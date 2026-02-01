import os, numpy as np, tqdm, pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from .model_utils import (
    Linear1DBlock, Variational1DBlock,
    ImgFeatureExtractor, VariationalImgFeatureExtractor,
    CompositeEncoder, VariationalCompositeEncoder
)

# Losses
class L1Loss(nn.Module):
    def forward(self, input, target):
        return torch.mean(torch.abs(input - target))

class CorrelationLoss(nn.Module):
    def __init__(self, eps=1e-5, weighted=False):
        super().__init__()
        self.eps = eps
        self.weighted = weighted

    def forward(self, input, target):
        vx = input - input.mean(dim=0, keepdims=True)
        vy = target - target.mean(dim=0, keepdims=True)
        pcc = torch.sum(vx * vy, dim=0, keepdims=True) / (
            torch.sqrt(torch.sum(vx ** 2, dim=0, keepdims=True)) *
            torch.sqrt(torch.sum(vy ** 2, dim=0, keepdims=True)) + self.eps
        )
        loss = 1 - pcc
        if self.weighted:
            loss = loss * target.sum(dim=0, keepdims=True)
        return loss.mean()

class HingeLoss(nn.Module):
    """
    Hinge GAN losses.

    Discriminator:
      L_D = E[relu(1 - D(real))] + E[relu(1 + D(fake))]
    Generator:
      L_G = -E[D(fake)]
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def disc_loss(real_output, fake_output):
        real_loss = torch.mean(F.relu(1.0 - real_output))
        fake_loss = torch.mean(F.relu(1.0 + fake_output))
        return real_loss + fake_loss

    @staticmethod
    def gen_loss(fake_output):
        return -torch.mean(fake_output)

class LeastSquaresLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_output, fake_output):
        real_loss = torch.mean((real_output - 1) ** 2)
        fake_loss = torch.mean(fake_output ** 2)
        return real_loss + fake_loss

def define_loss(loss_type):
    if loss_type == 'mse':
        return nn.MSELoss()
    if loss_type == 'l1':
        return L1Loss()
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    if loss_type == 'pearson':
        return CorrelationLoss()
    if loss_type == 'hinge':
        return HingeLoss()
    if loss_type == 'lsgan':
        return LeastSquaresLoss()
    raise ValueError(f"Unknown loss_type: {loss_type}")


# Trainer
class Trainer(nn.Module):
    def __init__(
        self,
        enc_source: nn.Module,
        dec_source: nn.Module,
        trans_s2t: nn.Module,
        disc_source: nn.Module,
        enc_target: nn.Module,
        dec_target: nn.Module,
        trans_t2s: nn.Module,
        disc_target: nn.Module,
        opts: dict,
        df_source: pd.DataFrame,  # used for reference in evaluation
    ):
        super().__init__()
        self.enc_source = enc_source
        self.dec_source = dec_source
        self.trans_s2t = trans_s2t
        self.disc_source = disc_source

        self.enc_target = enc_target
        self.dec_target = dec_target
        self.trans_t2s = trans_t2s
        self.disc_target = disc_target

        self.opts = opts
        self.device = opts['train_opt']['device']
        self.df_source = df_source.copy()

        self.use_amp = bool(self.opts['train_opt'].get("use_amp", True)) and torch.cuda.is_available()
        self.initialize_trainer()

    def initialize_trainer(self):
        train_opt = self.opts['train_opt']
        exp_opt = self.opts['exp_setting']
        self.log_dir = os.path.join(
            train_opt['log_dir'],
            f"fold_gene_{exp_opt['fold_gene']}",
            f"fold_sample_{exp_opt['fold_sample']}"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        for _, mod in self.named_modules():
            mod.to(self.device)
        for _, mod in self.named_modules():
            if hasattr(mod, 'reset_parameters'):
                mod.reset_parameters()

    def encode(self, x: dict, domain: str):
        return self.enc_source(x) if domain == 'source' else self.enc_target(x)

    def decode(self, z: torch.Tensor, domain: str):
        return self.dec_source(z) if domain == 'source' else self.dec_target(z)

    def forward(self, x: dict, domain: str):
        return self.decode(self.encode(x, domain), domain)

    def translate_s2t_latent(self, x: dict):
        return self.trans_s2t(self.enc_source(x))

    def translate_t2s_latent(self, x: dict):
        return self.trans_t2s(self.enc_target(x))

    def translate_s2t(self, x: dict):
        return self.dec_target(self.translate_s2t_latent(x))

    def translate_t2s(self, x: dict):
        return self.dec_source(self.translate_t2s_latent(x))

    def classify(self, z: torch.Tensor, domain: str):
        return self.disc_source(z) if domain == 'source' else self.disc_target(z)

    @staticmethod
    def _set_requires_grad(module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad_(flag)

    @torch.no_grad()
    def _eval_corr_streaming_selected_genes(
        self,
        dl_source,
        gene_names_target: list[str],
        eval_genes_union: list[str],
        eval_gene_idx: np.ndarray,     
        val_pos: np.ndarray,           
        test_pos: np.ndarray,         
        eval_max_cells: int | None,
        use_amp: bool,
    ):
        m = len(eval_genes_union)
        if m == 0 or eval_gene_idx.size == 0:
            return float("nan"), float("nan")

        sum_r  = np.zeros(m, dtype=np.float64)
        sum_p  = np.zeros(m, dtype=np.float64)
        sum_r2 = np.zeros(m, dtype=np.float64)
        sum_p2 = np.zeros(m, dtype=np.float64)
        sum_rp = np.zeros(m, dtype=np.float64)
        n_total = 0

        device_type = "cuda" if torch.cuda.is_available() and str(self.device).startswith("cuda") else "cpu"

        for batch in dl_source:
            # move tensors
            batch['input'] = batch['input'].to(self.device, non_blocking=True)
            if 'image' in batch:
                batch['image'] = batch['image'].to(self.device, non_blocking=True)

            cell_ids = batch['index']

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                out = self.translate_s2t(batch)
                out = out[:, eval_gene_idx]

            pred = out.float().cpu().numpy()
            real = self.df_source.loc[cell_ids, eval_genes_union].to_numpy(dtype=np.float32)

            b = real.shape[0]
            if eval_max_cells is not None and eval_max_cells > 0:
                if n_total >= eval_max_cells:
                    break
                if n_total + b > eval_max_cells:
                    keep = eval_max_cells - n_total
                    real = real[:keep]
                    pred = pred[:keep]
                    b = keep

            sum_r  += real.sum(axis=0)
            sum_p  += pred.sum(axis=0)
            sum_r2 += (real * real).sum(axis=0)
            sum_p2 += (pred * pred).sum(axis=0)
            sum_rp += (real * pred).sum(axis=0)
            n_total += b

            if eval_max_cells is not None and eval_max_cells > 0 and n_total >= eval_max_cells:
                break

        if n_total < 3:
            return float("nan"), float("nan")

        mean_r = sum_r / n_total
        mean_p = sum_p / n_total
        var_r  = sum_r2 / n_total - mean_r**2
        var_p  = sum_p2 / n_total - mean_p**2
        cov    = sum_rp / n_total - mean_r * mean_p

        denom = np.sqrt(np.maximum(var_r, 0.0)) * np.sqrt(np.maximum(var_p, 0.0))
        corr_all = np.where(denom > 0, cov / denom, np.nan)

        corr_val  = float(np.nanmean(corr_all[val_pos]))  if val_pos.size  else float("nan")
        corr_test = float(np.nanmean(corr_all[test_pos])) if test_pos.size else float("nan")
        return corr_val, corr_test

    def train_enc_dec(self, dict_dl: dict):
        os.makedirs(os.path.join(self.log_dir, 'enc_dec'), exist_ok=True)

        train_opt = self.opts['train_opt']
        use_amp = bool(train_opt.get("use_amp", True)) and torch.cuda.is_available()
        device_type = "cuda" if torch.cuda.is_available() and str(self.device).startswith("cuda") else "cpu"

        loss_fn = define_loss(train_opt['loss_type_enc_dec'])
        max_epochs = int(train_opt['epochs_enc_dec'])

        for domain in ['source', 'target']:
            print(f'Train Encoder Decoder for {domain} domain.')
            best_score = np.inf
            best_weight_enc = None
            best_weight_dec = None

            list_parameters = (
                list(self.enc_source.parameters()) + list(self.dec_source.parameters())
                if domain == 'source'
                else list(self.enc_target.parameters()) + list(self.dec_target.parameters())
            )
            optimizer = torch.optim.Adam(list_parameters)
            dl = dict_dl[domain]

            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            for epoch in range(max_epochs):
                if domain == 'source':
                    self.enc_source.train(); self.dec_source.train()
                else:
                    self.enc_target.train(); self.dec_target.train()

                loss_train = []
                for d in dl['train']:
                    optimizer.zero_grad(set_to_none=True)
                    d['input'] = d['input'].to(self.device, non_blocking=True)
                    if 'image' in d:
                        d['image'] = d['image'].to(self.device, non_blocking=True)

                    with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                        out = self.forward(d, domain)
                        loss = loss_fn(out, d['input'])

                    if use_amp and device_type == "cuda":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    loss_train.append(float(loss.item()))

                if domain == 'source' and 'val' in dl:
                    self.enc_source.eval(); self.dec_source.eval()
                    loss_val = []
                    with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=use_amp):
                        for d in dl['val']:
                            d['input'] = d['input'].to(self.device, non_blocking=True)
                            if 'image' in d:
                                d['image'] = d['image'].to(self.device, non_blocking=True)
                            out = self.forward(d, domain)
                            loss = loss_fn(out, d['input'])
                            loss_val.append(float(loss.item()))

                    val_mean = float(np.mean(loss_val) if len(loss_val) else np.nan)

                    if np.isfinite(val_mean) and val_mean < best_score:
                        best_score = val_mean
                        best_weight_enc = {k: v.detach().cpu().clone() for k, v in self.enc_source.state_dict().items()}
                        best_weight_dec = {k: v.detach().cpu().clone() for k, v in self.dec_source.state_dict().items()}
                        print(f'[enc_dec] best at epoch {epoch+1} val_loss={best_score:.4g}')

            if domain == 'source' and best_weight_enc is not None:
                self.enc_source.load_state_dict(best_weight_enc)
                self.dec_source.load_state_dict(best_weight_dec)

    def train(self, dict_dl: dict, load_best=True):
        gene_exp_val  = self.opts['exp_setting']['gene_names']['val']
        gene_exp_test = self.opts['exp_setting']['gene_names']['test']

        os.makedirs(os.path.join(self.log_dir, 'progress_log'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'weights'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'predictions'), exist_ok=True)

        pd.DataFrame({"gene": gene_exp_val}).to_csv(os.path.join(self.log_dir, "progress_log", "genes_val.csv"), index=False)
        pd.DataFrame({"gene": gene_exp_test}).to_csv(os.path.join(self.log_dir, "progress_log", "genes_test.csv"), index=False)

        train_opt = self.opts['train_opt']
        max_epochs = int(train_opt['epochs'])

        self.use_amp = bool(train_opt.get("use_amp", True)) and torch.cuda.is_available()
        device_type = "cuda" if torch.cuda.is_available() and str(self.device).startswith("cuda") else "cpu"
        scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and device_type == "cuda"))

        eval_every = int(train_opt.get("eval_every", 5))
        eval_max_cells = train_opt.get("eval_max_cells", 4096)
        eval_max_cells = None if eval_max_cells is None else int(eval_max_cells)

        print('Train translator')

        loss_fn_enc_dec = define_loss(train_opt['loss_type_enc_dec'])
        loss_fn_disc    = define_loss(train_opt['loss_type_disc'])
        loss_fn_cyc     = define_loss(train_opt['loss_type_cyc'])
        loss_fn_id      = define_loss(train_opt['loss_type_id'])

        gene_names_target = np.array(dict_dl['target']['train'].dataset.gene_names)
        input_genes_source = np.array(dict_dl['source']['train'].dataset.gene_exp_input)

        inter_gene_index_source = [np.where(input_genes_source == exp)[0][0] for exp in gene_exp_val]
        inter_gene_index_target = [np.where(gene_names_target == exp)[0][0] for exp in gene_exp_val]

        eval_genes_union = list(dict.fromkeys(list(gene_exp_val) + list(gene_exp_test)))
        g2i_t = {g: i for i, g in enumerate(gene_names_target.tolist())}

        eval_genes_union = [g for g in eval_genes_union if g in g2i_t]
        eval_gene_idx = np.array([g2i_t[g] for g in eval_genes_union], dtype=np.int64)

        u_pos = {g: i for i, g in enumerate(eval_genes_union)}
        val_pos  = np.array([u_pos[g] for g in gene_exp_val  if g in u_pos], dtype=np.int64)
        test_pos = np.array([u_pos[g] for g in gene_exp_test if g in u_pos], dtype=np.int64)

        optimizer_enc_dec = torch.optim.Adam(
            list(self.enc_source.parameters()) + list(self.dec_source.parameters()) +
            list(self.enc_target.parameters()) + list(self.dec_target.parameters()),
            weight_decay=0.001
        )
        optimizer_trans = torch.optim.Adam(
            list(self.trans_s2t.parameters()) + list(self.trans_t2s.parameters()),
            lr=train_opt['lr_trans'], weight_decay=0.001
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_trans, step_size=10, gamma=0.5)

        optimizer_disc = torch.optim.Adam(
            list(self.disc_source.parameters()) + list(self.disc_target.parameters()),
            lr=train_opt['lr_disc'], weight_decay=0.001
        )

        best_corr = -1e9
        best_epoch = None
        best_weight_enc_source = {k: v.detach().cpu().clone() for k, v in self.enc_source.state_dict().items()}
        best_weight_dec_target = {k: v.detach().cpu().clone() for k, v in self.dec_target.state_dict().items()}
        best_weight_trans_s2t  = {k: v.detach().cpu().clone() for k, v in self.trans_s2t.state_dict().items()}

        eval_history = []

        for epoch in tqdm.tqdm(range(max_epochs)):
            self.enc_source.train(); self.dec_target.train(); self.trans_s2t.train()
            self.enc_target.train(); self.dec_source.train(); self.trans_t2s.train()

            split = 'train'
            loader_length = max(len(dict_dl['source'][split]), len(dict_dl['target'][split]))
            dl_source_train = iter(dict_dl['source'][split])
            dl_target_train = iter(dict_dl['target'][split])

            for _ in range(loader_length):
                d_source = next(dl_source_train, None)
                d_target = next(dl_target_train, None)
                if d_source is None:
                    dl_source_train = iter(dict_dl['source'][split])
                    d_source = next(dl_source_train)
                if d_target is None:
                    dl_target_train = iter(dict_dl['target'][split])
                    d_target = next(dl_target_train)

                d_source['input'] = d_source['input'].to(self.device, non_blocking=True)
                d_target['input'] = d_target['input'].to(self.device, non_blocking=True)
                if 'image' in d_source:
                    d_source['image'] = d_source['image'].to(self.device, non_blocking=True)

                # schedules
                train_enc_dec = (not train_opt['fix_weight_enc_dec']) and (((epoch + 1) % train_opt['interval_enc_dec']) == 0)
                train_cyc     = ((epoch + 1) % train_opt['interval_cyc']) == 0
                train_id      = ((epoch + 1) % train_opt['interval_id']) == 0
                train_adv     = ((epoch + 1) % train_opt['interval_adv']) == 0
                train_disc    = ((epoch + 1) % train_opt['interval_disc']) == 0

                # encoder-decoder update (optional)
                if train_enc_dec:
                    optimizer_enc_dec.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                        out_source = self.forward(d_source, 'source')
                        out_target = self.forward(d_target, 'target')
                        loss_encdec = loss_fn_enc_dec(out_source, d_source['input']) + loss_fn_enc_dec(out_target, d_target['input'])

                    if self.use_amp and device_type == "cuda":
                        scaler.scale(loss_encdec).backward()
                        scaler.step(optimizer_enc_dec)
                        scaler.update()
                    else:
                        loss_encdec.backward()
                        optimizer_enc_dec.step()
                else:
                    with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                        _ = self.forward(d_source, 'source')
                        _ = self.forward(d_target, 'target')

                # discriminator update
                if train_disc:
                    self._set_requires_grad(self.disc_source, True)
                    self._set_requires_grad(self.disc_target, True)
                    optimizer_disc.zero_grad(set_to_none=True)

                    with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                        z_source = self.encode(d_source, 'source').detach()
                        z_target = self.encode(d_target, 'target').detach()
                        z_s2t    = self.trans_s2t(z_source).detach()
                        z_t2s    = self.trans_t2s(z_target).detach()

                        out_disc_source_real = self.classify(z_source, 'source')
                        out_disc_source_fake = self.classify(z_t2s, 'source')
                        out_disc_target_real = self.classify(z_target, 'target')
                        out_disc_target_fake = self.classify(z_s2t, 'target')

                        if train_opt['loss_type_disc'] == 'hinge':
                            loss_d_source = HingeLoss.disc_loss(out_disc_source_real, out_disc_source_fake)
                            loss_d_target = HingeLoss.disc_loss(out_disc_target_real, out_disc_target_fake)
                        else:
                            if isinstance(loss_fn_disc, nn.BCEWithLogitsLoss):
                                real_lbl_s = torch.ones_like(out_disc_source_real)
                                fake_lbl_s = torch.zeros_like(out_disc_source_fake)
                                real_lbl_t = torch.ones_like(out_disc_target_real)
                                fake_lbl_t = torch.zeros_like(out_disc_target_fake)
                                loss_d_source = loss_fn_disc(out_disc_source_real, real_lbl_s) + loss_fn_disc(out_disc_source_fake, fake_lbl_s)
                                loss_d_target = loss_fn_disc(out_disc_target_real, real_lbl_t) + loss_fn_disc(out_disc_target_fake, fake_lbl_t)
                            else:
                                loss_d_source = loss_fn_disc(out_disc_source_real, out_disc_source_fake)
                                loss_d_target = loss_fn_disc(out_disc_target_real, out_disc_target_fake)

                        loss_disc_total = train_opt['lambda_disc'] * (loss_d_source + loss_d_target)

                    if self.use_amp and device_type == "cuda":
                        scaler.scale(loss_disc_total).backward()
                        scaler.step(optimizer_disc)
                        scaler.update()
                    else:
                        loss_disc_total.backward()
                        optimizer_disc.step()

                # translator update
                if train_cyc or train_adv or train_id or train_mmd or train_coral:
                    self._set_requires_grad(self.disc_source, False)
                    self._set_requires_grad(self.disc_target, False)

                optimizer_trans.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    z_source = self.encode(d_source, 'source')
                    z_target = self.encode(d_target, 'target')
                    z_s2t    = self.trans_s2t(z_source)
                    z_t2s    = self.trans_t2s(z_target)

                    # Calculate losses
                    loss_adv = 0.0
                    if train_adv:
                        out_disc_s2t = self.classify(z_s2t, 'target')
                        out_disc_t2s = self.classify(z_t2s, 'source')
                        if train_opt['loss_type_disc'] == 'hinge':
                            loss_adv_s2t = HingeLoss.gen_loss(out_disc_s2t)
                            loss_adv_t2s = HingeLoss.gen_loss(out_disc_t2s)
                        else:
                            if isinstance(loss_fn_disc, nn.BCEWithLogitsLoss):
                                loss_adv_s2t = loss_fn_disc(out_disc_s2t, torch.ones_like(out_disc_s2t))
                                loss_adv_t2s = loss_fn_disc(out_disc_t2s, torch.ones_like(out_disc_t2s))
                            else:
                                loss_adv_s2t = -out_disc_s2t.mean()
                                loss_adv_t2s = -out_disc_t2s.mean()
                        loss_adv = train_opt['lambda_adv'] * (loss_adv_s2t + loss_adv_t2s)

                    loss_id = 0.0
                    if train_id:
                        out_s2t = self.decode(z_s2t, 'target')
                        out_t2s = self.decode(z_t2s, 'source')
                        loss_id = train_opt['lambda_id'] * (
                            loss_fn_id(out_s2t[:, inter_gene_index_target], d_source['input'][:, inter_gene_index_source]) +
                            loss_fn_id(out_t2s[:, inter_gene_index_source], d_target['input'][:, inter_gene_index_target])
                        )

                    loss_cyc = 0.0
                    if train_cyc:
                        z_s2t2s = self.trans_t2s(z_s2t)
                        z_t2s2t = self.trans_s2t(z_t2s)
                        out_s2t2s = self.decode(z_s2t2s, 'source')
                        out_t2s2t = self.decode(z_t2s2t, 'target')
                        loss_cyc = train_opt['lambda_cyc'] * (
                            loss_fn_cyc(out_s2t2s, d_source['input']) +
                            loss_fn_cyc(out_t2s2t, d_target['input'])
                        )

                    loss_total = loss_adv + loss_id + loss_cyc
                # Backpropagation and optimization steps
                if self.use_amp and device_type == "cuda":
                    scaler.scale(loss_total).backward()
                    scaler.step(optimizer_trans)
                    scaler.update()
                else:
                    loss_total.backward()
                    optimizer_trans.step()

            scheduler.step()

            do_eval = ((epoch + 1) % eval_every == 0) or (epoch == 0) or (epoch == max_epochs - 1)
            if do_eval:
                self.enc_source.eval(); self.dec_target.eval(); self.trans_s2t.eval()
                self.enc_target.eval(); self.dec_source.eval(); self.trans_t2s.eval()

                row = {"epoch": int(epoch + 1), "eval_max_cells": eval_max_cells}

                with torch.no_grad():
                    for split_sample in [s for s in ['train', 'val', 'test'] if s in dict_dl['source']]:
                        dl_eval = dict_dl['source'][split_sample]

                        corr_val, corr_test = self._eval_corr_streaming_selected_genes(
                            dl_source=dl_eval,
                            gene_names_target=gene_names_target.tolist(),
                            eval_genes_union=eval_genes_union,
                            eval_gene_idx=eval_gene_idx,
                            val_pos=val_pos,
                            test_pos=test_pos,
                            eval_max_cells=eval_max_cells,
                            use_amp=self.use_amp,
                        )

                        row[f"{split_sample}_corr_val"]  = float(corr_val)  if np.isfinite(corr_val)  else None
                        row[f"{split_sample}_corr_test"] = float(corr_test) if np.isfinite(corr_test) else None

                        if split_sample == 'val' and np.isfinite(corr_val) and corr_val > best_corr:
                            best_corr = float(corr_val)
                            best_epoch = int(epoch + 1)
                            best_weight_enc_source = {k: v.detach().cpu().clone() for k, v in self.enc_source.state_dict().items()}
                            best_weight_dec_target = {k: v.detach().cpu().clone() for k, v in self.dec_target.state_dict().items()}
                            best_weight_trans_s2t  = {k: v.detach().cpu().clone() for k, v in self.trans_s2t.state_dict().items()}

                eval_history.append(row)

                def _fmt(x):
                    return "NA" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.4f}"

                for split_sample in [s for s in ['val', 'test'] if s in dict_dl['source']]:
                    v = row.get(f"{split_sample}_corr_val", None)
                    t = row.get(f"{split_sample}_corr_test", None)
                    print(f"[eval][epoch {epoch+1}] split={split_sample:>5}  corr_val={_fmt(v)}  corr_test={_fmt(t)}")

                key_split = "val" if "val" in dict_dl["source"] else "train"
                v = row.get(f"{key_split}_corr_val", None)
                t = row.get(f"{key_split}_corr_test", None)
                print(f"(epoch {epoch+1}) do_eval=True  split_for_print={key_split}  corr_val={_fmt(v)}  corr_test={_fmt(t)}  best_corr_val={best_corr:.4f} best_epoch={best_epoch}")

        torch.save(best_weight_enc_source, os.path.join(self.log_dir, 'weights', 'enc_source.pth'))
        torch.save(best_weight_dec_target, os.path.join(self.log_dir, 'weights', 'dec_target.pth'))
        torch.save(best_weight_trans_s2t,  os.path.join(self.log_dir, 'weights', 'trans_s2t.pth'))

        
        # Save predictions at end
        self.enc_source.load_state_dict(best_weight_enc_source)
        self.dec_target.load_state_dict(best_weight_dec_target)
        self.trans_s2t.load_state_dict(best_weight_trans_s2t)
        self.enc_source.eval(); self.dec_target.eval(); self.trans_s2t.eval()

        val_idx = np.array([g2i_t[g] for g in gene_exp_val if g in g2i_t], dtype=np.int64)
        test_idx = np.array([g2i_t[g] for g in gene_exp_test if g in g2i_t], dtype=np.int64)
        val_cols = [g for g in gene_exp_val if g in g2i_t]
        test_cols = [g for g in gene_exp_test if g in g2i_t]

        def _get_loader_or_fallback(split_name: str):
            if split_name in dict_dl['source']:
                return dict_dl['source'][split_name]
            if 'val' in dict_dl['source']:
                return dict_dl['source']['val']
            return dict_dl['source']['train']

        @torch.no_grad()
        def _predict_split_valtest(dl):
            """
            Predict val/test gene sets for ALL cells in dl.
            Always returns DataFrames (never None).
            """
            ids = []
            preds_val = []
            preds_test = []

            for d in dl:
                d['input'] = d['input'].to(self.device, non_blocking=True)
                if 'image' in d:
                    d['image'] = d['image'].to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    out = self.translate_s2t(d)  # [B, G_target]

                out = out.float()
                if val_idx.size:
                    preds_val.append(out[:, val_idx].cpu().numpy())
                if test_idx.size:
                    preds_test.append(out[:, test_idx].cpu().numpy())
                ids.extend(list(d['index']))

            if val_idx.size and len(preds_val) > 0:
                df_val = pd.DataFrame(np.concatenate(preds_val, axis=0), index=ids, columns=val_cols).astype(np.float32)
            else:
                df_val = pd.DataFrame(index=ids, columns=val_cols, dtype=np.float32)

            if test_idx.size and len(preds_test) > 0:
                df_test = pd.DataFrame(np.concatenate(preds_test, axis=0), index=ids, columns=test_cols).astype(np.float32)
            else:
                df_test = pd.DataFrame(index=ids, columns=test_cols, dtype=np.float32)

            return df_val, df_test

        for split_sample in ['train', 'val', 'test']:
            dl_use = _get_loader_or_fallback(split_sample)
            df_val, df_test = _predict_split_valtest(dl_use)

            df_val.to_pickle(os.path.join(self.log_dir, 'predictions', f'best_pred_sample-{split_sample}_gene-val.pkl'))
            df_test.to_pickle(os.path.join(self.log_dir, 'predictions', f'best_pred_sample-{split_sample}_gene-test.pkl'))

        return None

def define_trainer(in_features_source, in_features_target, opts, df_source):
    model_opt = opts['model_opt']
    # Encoder for source
    latent_dim_source = model_opt['latent_dim_source']
    if model_opt['enc_type_source'] == '1d_simple':
        encoder_source = Linear1DBlock(in_features_source, latent_dim_source, model_opt['enc_features_source'], True, True)
    elif model_opt['enc_type_source'] == '1d_variational':
        encoder_source = Variational1DBlock(in_features_source, latent_dim_source, model_opt['enc_features_source'])
    else:
        in_channels = model_opt['in_channels_image']
        model_arch = model_opt['img_model_arch']
        pretrained = model_opt['img_model_pretrained']
        latent_dim_source = model_opt['latent_dim_img']
        if model_opt['enc_type_source'] == 'image_simple':
            encoder_source = ImgFeatureExtractor(in_channels, latent_dim_source, model_arch, pretrained)
        elif model_opt['enc_type_source'] == 'image_variational':
            encoder_source = VariationalImgFeatureExtractor(in_channels, latent_dim_source, model_arch, pretrained)
        else:
            latent_dim_source = model_opt['latent_dim_merge']
            merge_features = model_opt['merge_features']
            latent_dim_source_2d = model_opt['latent_dim_img']
            latent_dim_source_1d = model_opt['latent_dim_source']
            module_1d = Linear1DBlock(latent_dim_source_1d, in_features_source, model_opt['enc_features_source'], True, True)
            module_2d = ImgFeatureExtractor(in_channels, latent_dim_source_2d, model_arch, pretrained)
            if model_opt['enc_type_source'] == 'composite':
                encoder_source = CompositeEncoder(module_1d, module_2d, merge_features, latent_dim_source, True, True)
            elif model_opt['enc_type_source'] == 'composite_variational':
                encoder_source = VariationalCompositeEncoder(module_1d, module_2d, merge_features, latent_dim_source, True, True)
            else:
                raise ValueError("Unsupported enc_type_source")
    # Decoder for source
    decoder_source = Linear1DBlock(latent_dim_source, in_features_source, model_opt['dec_features_source'], False, False)
    # Discriminator for source
    disc_source = Linear1DBlock(latent_dim_source, 1, model_opt['disc_features_source'], False, False, 0.1)
    # Translator from source to target
    trans_s2t = Linear1DBlock(latent_dim_source, model_opt['latent_dim_target'], model_opt['trans_features_s2t'], True, True, 0.1)
    # Encoder for target
    if model_opt['enc_type_target'] == '1d_simple':
        encoder_target = Linear1DBlock(in_features_target, model_opt['latent_dim_target'], model_opt['enc_features_target'], True, True)
    elif model_opt['enc_type_target'] == '1d_variational':
        encoder_target = Variational1DBlock(in_features_target, model_opt['latent_dim_target'], model_opt['enc_features_target'])
    else:
        raise ValueError("Unsupported enc_type_target")
    # Decoder for target
    decoder_target = Linear1DBlock(model_opt['latent_dim_target'], in_features_target, model_opt['dec_features_target'], False, False)
    # Discriminator for target
    disc_target = Linear1DBlock(model_opt['latent_dim_target'], 1, model_opt['disc_features_target'], False, False, 0.1)
    # Translator from target to source
    trans_t2s = Linear1DBlock(model_opt['latent_dim_target'], latent_dim_source, model_opt['trans_features_t2s'], True, True, 0.1)
    # Define trainer
    return Trainer(
        encoder_source, decoder_source, trans_s2t, disc_source,
        encoder_target, decoder_target, trans_t2s, disc_target,
        opts, df_source
    )
