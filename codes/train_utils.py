import os, json, numpy as np, tqdm, yaml, pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import Linear1DBlock, Variational1DBlock, ImgFeatureExtractor, VariationalImgFeatureExtractor, CompositeEncoder, VariationalCompositeEncoder

class MMDLoss(nn.Module):
    def __init__(self, kernel='rbf', gamma=1.0):
        super(MMDLoss, self).__init__()
        self.kernel = kernel
        self.gamma = gamma

    def gaussian_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.view(x_size, 1, dim)
        y = y.view(1, y_size, dim)
        return torch.exp(-self.gamma * torch.sum((x - y) ** 2, dim=2))

    def forward(self, z_s, z_t):
        if self.kernel == 'rbf':
            K_ss = self.gaussian_kernel(z_s, z_s)
            K_tt = self.gaussian_kernel(z_t, z_t)
            K_st = self.gaussian_kernel(z_s, z_t)
            return K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
        else:
            raise NotImplementedError(f"Kernel '{self.kernel}' is not implemented in MMDLoss.")

class CorrelationLoss(nn.Module): 
    def __init__(
        self,
        eps = 1e-5,
        weighted = False
    ):
        super().__init__()
        self.eps = eps
        self.weighted = weighted
    def forward(self, input, target):
        vx = input - input.mean(dim = 0, keepdims = True)
        vy = target - target.mean(dim = 0, keepdims = True)
        pcc = torch.sum(vx * vy, dim = 0, keepdims = True) / (torch.sqrt(torch.sum(vx ** 2, dim = 0, keepdims = True)) * torch.sqrt(torch.sum(vy ** 2, dim = 0, keepdims = True)) + self.eps)
        loss = 1 - pcc
        if self.weighted:
            loss = loss * target.sum(dim = 0, keepdims = True)
        return loss.mean()

class CoralLoss(nn.Module):
    def __init__(self):
        super(CoralLoss, self).__init__()

    def forward(self, source_features, target_features):
        source_cov = self._covariance(source_features)
        target_cov = self._covariance(target_features)
        loss = torch.mean((source_cov - target_cov) ** 2)
        return loss

    def _covariance(self, features):
        batch_size = features.size(0)
        features_mean = torch.mean(features, dim=0)
        features_centered = features - features_mean
        cov = torch.mm(features_centered.t(), features_centered) / batch_size
        return cov

class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, real_output, fake_output):
        """
        Compute the hinge loss for GANs.
        
        Parameters:
        - real_output: Discriminator outputs for real samples (should ideally be close to 1).
        - fake_output: Discriminator outputs for fake samples (should ideally be close to 0).
        
        Returns:
        - The hinge loss for the discriminator.
        """
        real_loss = torch.mean(torch.relu(1.0 - real_output))
        fake_loss = torch.mean(torch.relu(1.0 + fake_output))
        return real_loss + fake_loss
    
class LeastSquaresLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, real_output, fake_output):
        """
        Compute the least squares loss for GANs.
        
        Parameters:
        - real_output: Discriminator outputs for real samples (should ideally be close to 1).
        - fake_output: Discriminator outputs for fake samples (should ideally be close to 0).
        
        Returns:
        - The least squares loss for the discriminator.
        """
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
    if loss_type == 'mmd':
        return MMDLoss(kernel='rbf', gamma=0.3)
    if loss_type == 'coral':
        return CoralLoss()
    if loss_type == 'hinge':
        return HingeLoss()
     

def define_simple_trainer(in_features_source, in_features_target, opts):
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
    # Decoder for source
    decoder_source = Linear1DBlock(latent_dim_source, in_features_source, model_opt['dec_features_source'], False, False)
    # Discriminator for source
    disc_source = Linear1DBlock(latent_dim_source, 1, model_opt['disc_features_source'], False, False)
    # Translator from source to target
    trans_s2t = Linear1DBlock(latent_dim_source, model_opt['latent_dim_target'], model_opt['trans_features_s2t'], True, True)
    # Encoder for target
    if model_opt['enc_type_target'] == '1d_simple':
        encoder_target = Linear1DBlock(in_features_target, model_opt['latent_dim_target'], model_opt['enc_features_target'], True, True)
    elif model_opt['enc_type_target'] == '1d_variational':
        encoder_target = Variational1DBlock(in_features_target, model_opt['latent_dim_target'], model_opt['enc_features_target'])
    # Decoder for target
    decoder_target = Linear1DBlock(model_opt['latent_dim_target'], in_features_target, model_opt['dec_features_target'], False, False)
    # Discriminator for target
    disc_target = Linear1DBlock(model_opt['latent_dim_target'], 1, model_opt['disc_features_target'], False, False)
    # Translator from target to source
    trans_t2s = Linear1DBlock(model_opt['latent_dim_target'], latent_dim_source, model_opt['trans_features_t2s'], True, True)
    # Define trainer
    trainer = SimpleTrainer(encoder_source, decoder_source, trans_s2t, disc_source, encoder_target, decoder_target, trans_t2s, disc_target)
    return trainer
    
class Trainer(nn.Module):
    def __init__(
        self,
        enc_source:nn.Module,
        dec_source:nn.Module,
        trans_s2t:nn.Module,
        disc_source:nn.Module,
        enc_target:nn.Module,
        dec_target:nn.Module,
        trans_t2s:nn.Module,
        disc_target:nn.Module,
        opts: dict,
        df_source: pd.DataFrame, # used for reference in evaluation step
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
        self.initialize_trainer()
        self.df_source = df_source.copy()
    def initialize_trainer(self):
        """
        make (1) log, (2) device, (3) initialize weights, (4) make progress history
        """
        train_opt = self.opts['train_opt']
        exp_opt = self.opts['exp_setting']
        self.log_dir = os.path.join(train_opt['log_dir'], f"fold_gene_{exp_opt['fold_gene']}", f"fold_sample_{exp_opt['fold_sample']}")
        os.makedirs(self.log_dir, exist_ok = True)
       
        for name, mod in self.named_modules():
            mod.to(self.device)
        for name, mod in self.named_modules():
            if hasattr(mod, 'reset_parameters'):
                mod.reset_parameters()
    def init_weight(self):
        for name, mod in self.named_modules():
            if hasattr(mod, 'reset_parameters'):
                mod.reset_parameters()
            if isinstance(mod, nn.Linear):
                mod.weight.data.normal_(mean = 0.0, std = 1.0)
                if mod.bias is not None:
                    mod.bias.data.zero_()
    def encode_source(self, x:dict):
        return self.enc_source(x)
    def encode_target(self, x:dict):
        return self.enc_target(x)
    def encode(self, x:dict, domain: str):
        if domain == 'source':
            return self.encode_source(x)
        else:
            return self.encode_target(x)
    def decode_source(self, z):
        return self.dec_source(z)
    def decode_target(self, z):
        return self.dec_target(z)
    def decode(self, x:dict, domain: str):
        if domain == 'source':
            return self.decode_source(x)
        else:
            return self.decode_target(x)
    def forward_source(self, x:dict):
        return self.dec_source(self.enc_source(x))
    def forward_target(self, x:dict):
        return self.dec_target(self.enc_target(x))
    def forward(self, x:dict, domain:str):
        if domain == 'source':
            return self.forward_source(x)
        else:
            return self.forward_target(x)
    def translate_s2t_latent(self, x:dict):
        return self.trans_s2t(self.enc_source(x))
    def translate_t2s_latent(self, x:dict):
        return self.trans_t2s(self.enc_target(x))
    def translate_s2t(self, x:dict):
        return self.dec_target(self.translate_s2t_latent(x))
    def classify_source(self, z:torch.Tensor):
        return self.disc_source(z)
    def classify_target(self, z:torch.Tensor):
        return self.disc_target(z)
    def classify(self, z:torch.Tensor, domain: str):
        if domain == 'source':
            return self.classify_source(z)
        else:
            return self.classify_target(z)        
    def train_enc_dec(self, dict_dl: dict):
        os.makedirs(os.path.join(self.log_dir, 'enc_dec'), exist_ok = True)
        history = {
            domain: {
                split: {'mse_avg': [], 'corr_avg': [], 'mse': [], 'corr': [], 'loss': []}
                for split in ['train', 'val']
            }
            for domain in ['source', 'target']
        }
        loss_fn = define_loss(self.opts['train_opt']['loss_type_enc_dec'])
        max_epochs = self.opts['train_opt']['epochs_enc_dec']
        for domain in ['source', 'target']:
            print(f'Train Encoder Decoder for {domain} domain.')
            best_score = -1
            best_epoch = None
            best_weight_enc = None
            best_weight_dec = None
            list_parameters = list(self.enc_source.parameters()) + list(self.dec_source.parameters()) if domain == 'source' else list(self.enc_target.parameters()) + list(self.dec_target.parameters())
            optimizer = torch.optim.Adam(list_parameters)
            dl = dict_dl[domain]
            for epoch in range(max_epochs):
                record = {
                    split:{
                        metric: []
                        for metric in ['loss', 'real', 'pred']
                    }
                    for split in [k for k in dl.keys() if k != 'test']
                }
                # Train
                split = 'train'
                if domain == 'source':
                    self.enc_source.train(); self.dec_source.train()
                else:
                    self.enc_target.train(); self.dec_target.train()
                
                for d in dl[split]:
                    optimizer.zero_grad()
                    d['input'] = d['input'].to(self.device)
                    if 'image' in d:
                        d['image'] = d['image'].to(self.device)
                    out = self.forward(d, domain)
                    loss = loss_fn(out, d['input'])
                    loss.backward()
                    optimizer.step()
                    # record progress
                    record[split]['loss'].append(loss.item())
                    record[split]['pred'].append(out.detach().cpu().numpy())
                    record[split]['real'].append(d['input'].cpu().numpy())
                # Evaluation
                if domain == 'source':
                    split = 'val'
                    self.enc_source.eval(); self.dec_source.eval()
                    
                    with torch.no_grad():
                        for d in dl[split]:
                            d['input'] = d['input'].to(self.device)
                            if 'image' in d:
                                d['image'] = d['image'].to(self.device)
                            out = self.forward(d, domain)
                            loss = loss_fn(out, d['input'])
                            # record progress
                            record[split]['loss'].append(loss.item())
                            record[split]['pred'].append(out.detach().cpu().numpy())
                            record[split]['real'].append(d['input'].cpu().numpy())
                # record progress (correlation)
                for split in ['train', 'val']:
                    if split in record:
                        pred = np.concatenate(record[split]['pred'])
                        real = np.concatenate(record[split]['real'])
                        
                        # Check if all zeros or variance issues
                        if np.all(pred == 0) or np.all(real == 0):
                            print(f"Warning: All zeros in {split} data for domain {domain}")
        
                        mse = [float(np.mean((r-p)**2)) for r,p in zip(real.T, pred.T)]
                        corr = [float(np.corrcoef(r,p)[0,1]) for r,p in zip(real.T, pred.T)]
                        loss = np.mean(record[split]['loss'])
                        history[domain][split]['loss'].append(float(loss))
                        history[domain][split]['corr'].append(corr)
                        history[domain][split]['mse'].append(mse)
                # save best parameter
                if domain == 'source' and best_score < np.mean(history['source']['val']['corr'][-1]):
                    best_score = np.mean(history['source']['val']['corr'][-1])
                    best_epoch = epoch + 1
                    best_weight_enc = self.enc_source.state_dict() if domain == 'source' else self.enc_target.state_dict()
                    best_weight_dec = self.dec_source.state_dict() if domain == 'source' else self.dec_target.state_dict()
                    print(f'best model selected at {epoch} with {best_score:.3f}')
            if domain == 'source':
                print(f'Loading best weights at epoch {best_epoch} with corr {best_score}')
                self.enc_source.load_state_dict(best_weight_enc)
                self.dec_source.load_state_dict(best_weight_dec)
        json.dump(history, open(os.path.join(self.log_dir, 'enc_dec', 'history.json'), 'w'))
    
    def train(self, dict_dl: dict, load_best = True):
        gene_exp_val = self.opts['exp_setting']['gene_names']['val']
        gene_exp_test = self.opts['exp_setting']['gene_names']['test']
        os.makedirs(os.path.join(self.log_dir, 'progress_log'), exist_ok = True)
        os.makedirs(os.path.join(self.log_dir, 'weights'), exist_ok = True)
        os.makedirs(os.path.join(self.log_dir, 'predictions'), exist_ok = True)
        train_opt = self.opts['train_opt']
        max_epochs = train_opt['epochs']
        print('Train translator')
        # Define progress record dictionary
        history = {}
        for split in ['train', 'val', 'test']:
            history[split] = {}
            if split == 'source':
                history[split]['source'] = {
                    'corr_val': [],
                    'corr_test': []
                }
            else:
                for domain in ['source', 'target']:
                    if split != 'train':
                        history[split][domain] = {
                            'corr_val': [],
                            'corr_test': []
                        }
                    else:
                        history[split][domain] = {
                            'corr_val': [],
                            'corr_test': [],
                            'disc_pred_real': [],
                            'disc_pred_fake': [],
                            'disc_loss_real': [],
                            'disc_loss_fake': [],
                            'enc_dec_loss': [],
                            'loss_adv': [],
                            'loss_cyc': [],
                            'loss_id': [],
                            'loss_div': [],
                            'loss_mmd': [],
                            'loss_coral': [],
                            'loss_nn': []
                        }
        best_pred = {
            split_sample: {
                split_gene: None
                for split_gene in ['val', 'test']
            }
            for split_sample in ['train', 'val', 'test']
        }
        # define loss functions
        loss_fn_enc_dec = define_loss(train_opt['loss_type_enc_dec'])
        loss_fn_disc = define_loss(train_opt['loss_type_disc'])
        loss_fn_cyc = define_loss(train_opt['loss_type_cyc'])
        loss_fn_id = define_loss(train_opt['loss_type_id'])
        loss_fn_mmd = define_loss(train_opt['loss_type_mmd'])
        loss_fn_coral = define_loss(train_opt['loss_type_coral'])

        # Find index for validation/test gene expressions
        # validation gene expressions are used for id loss when training translator
        # test gene expressions are used for evaluating model
        gene_names_source = np.array(dict_dl['source']['train'].dataset.gene_names)
        gene_names_target = np.array(dict_dl['target']['train'].dataset.gene_names)
        input_genes_source = np.array(dict_dl['source']['train'].dataset.gene_exp_input)
        inter_gene_index_source = [np.where(input_genes_source == exp)[0][0] for exp in gene_exp_val]
        inter_gene_index_target = [np.where(gene_names_target == exp)[0][0] for exp in gene_exp_val]
        eval_gene_index_source = [np.where(gene_names_source == exp)[0][0] for exp in gene_exp_test]
        eval_gene_index_target = [np.where(gene_names_target == exp)[0][0] for exp in gene_exp_test]
        # Define optimizers - fix
        import torch
        optimizer_enc_dec = torch.optim.Adam(list(self.enc_source.parameters()) + list(self.dec_source.parameters()) + list(self.enc_target.parameters()) + list(self.dec_target.parameters()), weight_decay=0.001)

        optimizer_trans = torch.optim.Adam(list(self.trans_s2t.parameters()) + list(self.trans_t2s.parameters()), lr=train_opt['lr_trans'], weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_trans, step_size=10, gamma=0.5)
        optimizer_disc = torch.optim.Adam(list(self.disc_source.parameters()) + list(self.disc_target.parameters()), lr=train_opt['lr_disc'], weight_decay=0.001)
        # keep best metrics for model selection
        best_corr = -1
        best_weight_enc_source = self.enc_source.state_dict()
        best_weight_dec_source = self.dec_source.state_dict()
        best_weight_trans_s2t = self.trans_s2t.state_dict()
        
        loss_adv_s2t_list = []
        loss_adv_t2s_list = []
        loss_disc_source_real_list = []
        loss_disc_target_real_list = []
        loss_disc_source_fake_list = []
        loss_disc_target_fake_list = []
        acc_disc_source_real_list = []
        acc_disc_source_fake_list = []
        acc_disc_target_real_list = []
        acc_disc_target_fake_list = []

        from tqdm import tqdm
        for epoch in tqdm(range(max_epochs)):
            # define progress for current epoch
            record = {
                'train': {
                    domain: {
                        'enc_dec': {'loss': []},
                        'disc': {'loss_real': [], 'loss_fake': [], 'pred_real': [], 'pred_fake': []},
                        'trans': {'loss_adv': [], 'loss_cyc': [], 'loss_id': [], 'loss_div': [], 'loss_mmd': [], 'loss_coral' :[], 'loss_nn': [], 'pred': []}
                    }
                    for domain in ['source', 'target']
                }
            }
            record['train']['gene_s2t'] = []
            record['train']['cell_id_source'] = []
            record['val'] = {'gene_s2t': [], 'cell_id_source': []}
            record['test'] = {'gene_s2t': [], 'cell_id_source': []}
            progress = {
                split_sample: {
                        'corr_val': [],
                        'corr_test': [],
                } 
                for split_sample in ['train', 'val', 'test']
            }
            """
            Train
                1. load mini-batch data
                2. train encoder-decoder
                3. train discriminator
                4. train translator
                    4-1. adversarial loss
                    4-2. cycle loss
                    4-3. id loss
                5. Summarize progress
            """
            self.enc_source.train(); self.dec_target.train(); self.trans_s2t.train()
            self.enc_target.train(); self.dec_source.train(); self.trans_t2s.train()
            split = 'train'
            
            loader_length = max(len(dict_dl['source'][split]), len(dict_dl['target'][split]))
            dl_source_train = iter(dict_dl['source'][split])
            dl_target_train = iter(dict_dl['target'][split])
            draw_umap = True
            for _ in range(loader_length):
                """ 1. Load mini-batch data """
                d_source = next(dl_source_train, None)
                d_target = next(dl_target_train, None)
                if d_source is None:
                    dl_source_train = iter(dict_dl['source'][split])
                    d_source = next(dl_source_train, None)
                if d_target is None:
                    dl_target_train = iter(dict_dl['target'][split])
                    d_target = next(dl_target_train, None)                    
                d_source['input'] = d_source['input'].to(self.device)
                d_target['input'] = d_target['input'].to(self.device)
                
                if 'image' in d_source:
                    d_source['image'] = d_source['image'].to(self.device)
                z_source = self.encode(d_source, 'source')
                z_target = self.encode(d_target, 'target')
                z_s2t = self.trans_s2t(z_source)
                z_t2s = self.trans_t2s(z_target)
                z_s2t2s = self.trans_t2s(z_s2t)
                z_t2s2t = self.trans_s2t(z_t2s)
                # weight update or not
                train_enc_dec = (not train_opt['fix_weight_enc_dec']) and (((epoch+1) % train_opt['interval_enc_dec']) == 0)
                train_cyc = ((epoch+1) % train_opt['interval_cyc']) == 0
                train_id = ((epoch+1) % train_opt['interval_id']) == 0
                train_adv = ((epoch+1) % train_opt['interval_adv']) == 0
                train_disc = ((epoch+1) % train_opt['interval_disc']) == 0
                train_mmd = ((epoch + 1) % train_opt['interval_mmd']) == 0
                train_coral = ((epoch + 1) % train_opt['interval_coral']) == 0

                """ 2. train encoder-decoder """
                if train_enc_dec:
                    for p_groups in optimizer_enc_dec.param_groups:
                        for p in p_groups['params']:
                            p.requires_grad = True
                    optimizer_enc_dec.zero_grad()
                    out_source = self.forward(d_source, 'source')
                    out_target = self.forward(d_target, 'target')
                else:
                    with torch.no_grad():
                        out_source = self.forward(d_source, 'source')
                        out_target = self.forward(d_target, 'target')
                loss_encdec_source = loss_fn_enc_dec(out_source, d_source['input'])
                loss_encdec_target = loss_fn_enc_dec(out_target, d_target['input'])
                if train_enc_dec:
                    loss = loss_encdec_source + loss_encdec_target
                    loss.backward()
                    optimizer_enc_dec.step()
                # record loss and prediction for encoder-decoder
                record[split]['source']['enc_dec']['loss'].append(float(loss_encdec_source.item()))
                record[split]['target']['enc_dec']['loss'].append(float(loss_encdec_target.item()))
                for p_groups in optimizer_enc_dec.param_groups:
                    for p in p_groups['params']:
                        p.requires_grad = False
                """ 3. Train discriminator """

                import torch
                from torch.autograd import grad

                def compute_gradient_penalty(D, real_samples, fake_samples, domain, device=self.device):
                    """Calculates the gradient penalty for WGAN-GP"""
                    # Random weight term for interpolation between real and fake samples
                    alpha = torch.rand(real_samples.size(0), 1, device=device)
                    # Interpolate between real and fake samples
                    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

                    d_interpolates = D(interpolates, domain)
                    
                    # Calculate gradients of discriminator outputs w.r.t. interpolated inputs
                    gradients = grad(
                        outputs=d_interpolates,
                        inputs=interpolates,
                        grad_outputs=torch.ones_like(d_interpolates),
                        create_graph=True, retain_graph=True, only_inputs=True
                    )[0]
                    
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

                    return gradient_penalty

                if train_disc:
                    for p_groups in optimizer_disc.param_groups:
                        for p in p_groups['params']:
                            p.requires_grad = True
                    optimizer_disc.zero_grad()
                    z_source = self.encode(d_source, 'source')
                    z_target = self.encode(d_target, 'target')
                    z_s2t = self.trans_s2t(z_source)
                    z_t2s = self.trans_t2s(z_target)
                    out_disc_source = self.classify(z_source, 'source')
                    out_disc_target = self.classify(z_target, 'target')
                    out_disc_t2s = self.classify(z_t2s, 'source')
                    out_disc_s2t = self.classify(z_s2t, 'target')
                else:
                    with torch.no_grad():
                        z_source = self.encode(d_source, 'source')
                        z_target = self.encode(d_target, 'target')
                        z_s2t = self.trans_s2t(z_source)
                        z_t2s = self.trans_t2s(z_target)
                        out_disc_source = self.classify(z_source, 'source')
                        out_disc_target = self.classify(z_target, 'target')
                        out_disc_t2s = self.classify(z_t2s, 'source')
                        out_disc_s2t = self.classify(z_s2t, 'target')

                real_labels_source = torch.ones_like(out_disc_source)
                fake_labels_source = torch.zeros_like(out_disc_t2s)
                real_labels_target = torch.ones_like(out_disc_target)
                fake_labels_target = torch.zeros_like(out_disc_s2t)

                def calculate_accuracy(predictions, labels, threshold=0.5, real=True):
                    """
                    Calculate accuracy given predictions and true labels.
                    
                    For real samples, the prediction should be greater than threshold to be considered correct.
                    For fake samples, the prediction should be less than threshold to be considered correct.
                    """
                    if real:
                        # For real samples, we expect predictions >= threshold (indicating real).
                        correct_predictions = (predictions >= threshold).float() == (labels == 1.0).float()
                    else:
                        # For fake samples, we expect predictions < threshold (indicating fake).
                        correct_predictions = (predictions < threshold).float() == (labels == 0.0).float()
                    
                    accuracy = correct_predictions.float().mean()
                    return accuracy
                
                # Real source labels
                loss_disc_source_real = loss_fn_disc(out_disc_source, real_labels_source)
                # Fake source-to-target labels
                loss_disc_source_fake = loss_fn_disc(out_disc_t2s, fake_labels_source)

                # Real target labels
                loss_disc_target_real = loss_fn_disc(out_disc_target, real_labels_target)
                # Fake target-to-source labels
                loss_disc_target_fake = loss_fn_disc(out_disc_s2t, fake_labels_target)


                loss_disc_target_fake_list.append(float(loss_disc_target_fake.item()))
                loss_disc_source_fake_list.append(float(loss_disc_source_fake.item()))
                loss_disc_target_real_list.append(float(loss_disc_target_real.item()))
                loss_disc_source_real_list.append(float(loss_disc_source_real.item()))
               
                # Calculate losses
                loss_disc_source = (loss_disc_source_real + loss_disc_source_fake) * train_opt['lambda_disc']
                loss_disc_target = (loss_disc_target_real + loss_disc_target_fake) * train_opt['lambda_disc']

                # Backpropagation and optimization steps
                if train_disc:
                    loss = loss_disc_source + loss_disc_target
                    lambda_gp = 10  # Weight for gradient penalty, can tune this value
                    gradient_penalty_source = compute_gradient_penalty(self.classify, z_source, z_t2s, 'source')
                    gradient_penalty_target = compute_gradient_penalty(self.classify, z_target, z_s2t, 'target')

                    loss += lambda_gp * (gradient_penalty_source + gradient_penalty_target)
                    loss.backward()
                    optimizer_disc.step()

                # record loss and prediction for discriminator
                record[split]['source']['disc']['loss_real'].append(float(loss_disc_source_real.item()))
                record[split]['source']['disc']['loss_fake'].append(float(loss_disc_source_fake.item()))
                record[split]['target']['disc']['loss_real'].append(float(loss_disc_target_real.item()))
                record[split]['target']['disc']['loss_fake'].append(float(loss_disc_target_fake.item()))
                record[split]['source']['disc']['pred_real'].append(out_disc_source.detach().cpu().numpy())
                record[split]['source']['disc']['pred_fake'].append(out_disc_t2s.detach().cpu().numpy())
                record[split]['target']['disc']['pred_real'].append(out_disc_target.detach().cpu().numpy())
                record[split]['target']['disc']['pred_fake'].append(out_disc_s2t.detach().cpu().numpy())
                
                # Calculate accuracy
                acc_disc_source_real = calculate_accuracy(out_disc_source, real_labels_source, real=True)
                acc_disc_source_fake = calculate_accuracy(out_disc_t2s, fake_labels_source, real=False)
                acc_disc_target_real = calculate_accuracy(out_disc_target, real_labels_target, real=True)
                acc_disc_target_fake = calculate_accuracy(out_disc_s2t, fake_labels_target, real=False)

                # Store accuracy values
                acc_disc_source_real_list.append(float(acc_disc_source_real))
                acc_disc_source_fake_list.append(float(acc_disc_source_fake))
                acc_disc_target_real_list.append(float(acc_disc_target_real))
                acc_disc_target_fake_list.append(float(acc_disc_target_fake))


                """ 4. train translator """
                if train_cyc or train_adv or train_id or train_mmd:
                    for p_groups in optimizer_disc.param_groups:
                        for p in p_groups['params']:
                            p.requires_grad = False
                optimizer_trans.zero_grad()
                
                if train_mmd:
                    z_source = self.encode(d_source, 'source').detach()
                    z_target = self.encode(d_target, 'target').detach()
                    z_s2t = self.trans_s2t(z_source)
                    z_t2s = self.trans_t2s(z_target)
                    loss_mmd_s2t = loss_fn_mmd(z_s2t, z_target)
                    loss_mmd_t2s = loss_fn_mmd(z_t2s, z_source)
                    loss_mmd = (loss_mmd_s2t + loss_mmd_t2s) * train_opt['lambda_mmd']
                else:
                    with torch.no_grad():
                        z_source = self.encode(d_source, 'source').detach()
                        z_target = self.encode(d_target, 'target').detach()
                        z_s2t = self.trans_s2t(z_source)
                        z_t2s = self.trans_t2s(z_target)
                        loss_mmd_s2t = loss_fn_mmd(z_s2t, z_target)
                        loss_mmd_t2s = loss_fn_mmd(z_t2s, z_source)
                        loss_mmd = (loss_mmd_s2t + loss_mmd_t2s) * train_opt['lambda_mmd']
               
                if train_coral:
                    z_source = self.encode(d_source, 'source')
                    z_target = self.encode(d_target, 'target')
                    z_s2t = self.trans_s2t(z_source)
                    z_t2s = self.trans_t2s(z_target)
                    loss_coral_s2t = loss_fn_coral(z_s2t, z_target)
                    loss_coral_t2s = loss_fn_coral(z_t2s, z_source)
                    loss_coral = (loss_coral_s2t + loss_coral_t2s) * train_opt['lambda_coral']
                else:
                    with torch.no_grad():
                        z_source = self.encode(d_source, 'source')
                        z_target = self.encode(d_target, 'target')
                        z_s2t = self.trans_s2t(z_source)
                        z_t2s = self.trans_t2s(z_target)
                        loss_coral_s2t = loss_fn_coral(z_s2t, z_target)
                        loss_coral_t2s = loss_fn_coral(z_t2s, z_source)
                        loss_coral = (loss_coral_s2t + loss_coral_t2s) * train_opt['lambda_coral']

                if train_adv:
                    z_source = self.encode(d_source, 'source')
                    z_target = self.encode(d_target, 'target')
                    z_s2t = self.trans_s2t(z_source)
                    z_t2s = self.trans_t2s(z_target)
                    out_disc_t2s = self.classify(z_t2s, 'source')
                    out_disc_s2t = self.classify(z_s2t, 'target')
                    loss_adv_s2t = loss_fn_disc(out_disc_s2t, torch.ones_like(out_disc_s2t))
                    loss_adv_t2s = loss_fn_disc(out_disc_t2s, torch.ones_like(out_disc_t2s))
                    loss_adv = (loss_adv_s2t + loss_adv_t2s) * train_opt['lambda_adv']
                    loss_adv_s2t_list.append(float(loss_adv_s2t.item()))
                    loss_adv_t2s_list.append(float(loss_adv_t2s.item()))
                else:
                    with torch.no_grad():
                        z_source = self.encode(d_source, 'source')
                        z_target = self.encode(d_target, 'target')
                        z_s2t = self.trans_s2t(z_source)
                        z_t2s = self.trans_t2s(z_target)
                        out_disc_t2s = self.classify(z_t2s, 'source')
                        out_disc_s2t = self.classify(z_s2t, 'target')
                        loss_adv_s2t = loss_fn_disc(out_disc_s2t, torch.ones_like(out_disc_s2t))
                        loss_adv_t2s = loss_fn_disc(out_disc_t2s, torch.ones_like(out_disc_t2s))
                if train_id:
                    z_source = self.encode(d_source, 'source')
                    z_target = self.encode(d_target, 'target')
                    z_s2t = self.trans_s2t(z_source)
                    z_t2s = self.trans_t2s(z_target)
                    out_s2t = self.decode(z_s2t, 'target')
                    out_t2s = self.decode(z_t2s, 'source')
                    loss_id_s2t = loss_fn_id(out_s2t[:, inter_gene_index_target], d_source['input'][:, inter_gene_index_source])
                    loss_id_t2s = loss_fn_id(out_t2s[:, inter_gene_index_source], d_target['input'][:, inter_gene_index_target])
                    loss_id = (loss_id_s2t + loss_id_t2s) * train_opt['lambda_id']
                else:
                    with torch.no_grad():
                        z_source = self.encode(d_source, 'source')
                        z_target = self.encode(d_target, 'target')
                        z_s2t = self.trans_s2t(z_source)
                        z_t2s = self.trans_t2s(z_target)
                        out_s2t = self.decode(z_s2t, 'target')
                        out_t2s = self.decode(z_t2s, 'source')
                        loss_id_s2t = loss_fn_id(out_s2t[:, inter_gene_index_target], d_source['input'][:, inter_gene_index_source])
                        loss_id_t2s = loss_fn_id(out_t2s[:, inter_gene_index_source], d_target['input'][:, inter_gene_index_target])
                if train_cyc:
                    z_source = self.encode(d_source, 'source')
                    z_target = self.encode(d_target, 'target')
                    z_s2t = self.trans_s2t(z_source)
                    z_t2s = self.trans_t2s(z_target)
                    z_s2t2s = self.trans_t2s(z_s2t)
                    z_t2s2t = self.trans_s2t(z_t2s)
                    out_s2t2s = self.decode(z_s2t2s, 'source')
                    out_t2s2t = self.decode(z_t2s2t, 'target')
                    loss_cyc_source = loss_fn_cyc(out_s2t2s, d_source['input'])
                    loss_cyc_target = loss_fn_cyc(out_t2s2t, d_target['input'])
                    loss_cyc = (loss_cyc_source + loss_cyc_target) * train_opt['lambda_cyc']
                else:
                    with torch.no_grad():
                        z_source = self.encode(d_source, 'source')
                        z_target = self.encode(d_target, 'target')
                        z_s2t = self.trans_s2t(z_source)
                        z_t2s = self.trans_t2s(z_target)
                        z_s2t2s = self.trans_t2s(z_s2t)
                        z_t2s2t = self.trans_s2t(z_t2s)
                        out_s2t2s = self.decode(z_s2t2s, 'source')
                        out_t2s2t = self.decode(z_t2s2t, 'target')
                        loss_cyc_source = loss_fn_cyc(out_s2t2s, d_source['input'])
                        loss_cyc_target = loss_fn_cyc(out_t2s2t, d_target['input'])
                
                loss =  loss_id + loss_cyc + loss_adv + loss_coral + loss_mmd

                loss.backward()
                optimizer_trans.step()
                # record loss and prediction for translator
                record[split]['source']['trans']['loss_adv'].append(float(loss_adv_s2t.item()))
                record[split]['source']['trans']['loss_cyc'].append(float(loss_cyc_source.item()))
                record[split]['source']['trans']['loss_id'].append(float(loss_id_s2t.item()))
                record[split]['source']['trans']['loss_mmd'].append(float(loss_mmd_s2t.item()))
                record[split]['source']['trans']['loss_coral'].append(float(loss_coral_s2t.item()))
                record[split]['target']['trans']['loss_adv'].append(float(loss_adv_t2s.item()))
                record[split]['target']['trans']['loss_cyc'].append(float(loss_cyc_target.item()))
                record[split]['target']['trans']['loss_id'].append(float(loss_id_t2s.item()))
                record[split]['target']['trans']['loss_mmd'].append(float(loss_mmd_t2s.item()))
                record[split]['target']['trans']['loss_coral'].append(float(loss_coral_t2s.item()))

            scheduler.step()
            
            """
            Validation & test - only record real and prediction to check the correlation
            """
            self.enc_source.eval(); self.dec_target.eval(); self.trans_s2t.eval()
            self.enc_target.eval(); self.dec_source.eval(); self.trans_t2s.eval()
            with torch.no_grad():
                for split in [s for s in ['train', 'val', 'test'] if s in dict_dl['source']]:  
                    for d_source in dict_dl['source'][split]:
                        d_source['input'] = d_source['input'].to(self.device)
                        if 'image' in d_source:
                            d_source['image'] = d_source['image'].to(self.device)
                        out_s2t = self.translate_s2t(d_source)
                        record[split]['gene_s2t'].append(out_s2t.detach().cpu().numpy())
                        record[split]['cell_id_source'].append(d_source['index'])
            """
            5. Summarize progress for current epoch
            """
            is_curr_best = False
            for split_sample in [s for s in ['val', 'train', 'test'] if s in dict_dl['source']]:
                # Get Correlation for synthetic data
                cell_id_source = np.concatenate(record[split_sample]['cell_id_source'])
                df_pred = pd.DataFrame(data = np.concatenate(record[split_sample]['gene_s2t']), index = cell_id_source, columns = gene_names_target)
                df_real = self.df_source.loc[cell_id_source].copy()
                for gene_names, split_gene in zip([gene_exp_val, gene_exp_test], ['val', 'test']):
                    if split_sample not in dict_dl['source']:
                        progress[split_sample][f'corr_{split_gene}'] = progress['val'][f'corr_{split_gene}']
                        continue
                    list_corr = []
                    for gene_name in gene_names:
                        p = df_pred.loc[:, gene_name].values
                        r = df_real.loc[:, gene_name].values
                        list_corr.append(np.corrcoef(r,p)[0,1])
                    progress[split_sample][f'corr_{split_gene}'] = np.mean(list_corr)
                    history[split_sample]['source'][f'corr_{split_gene}'].append(np.mean(list_corr))
                    if split_sample == 'val' and best_corr < progress['val']['corr_val']:
                        best_corr = float(progress[split_sample][f'corr_val'])
                        best_weight_enc_source = self.enc_source.state_dict().copy()
                        best_weight_dec_target = self.dec_target.state_dict().copy()
                        best_weight_trans_s2t = self.trans_s2t.state_dict().copy()
                        is_curr_best = True
                        print('Best correlation found.')
                    if is_curr_best:
                        best_pred[split_sample][split_gene] = df_pred.astype(np.float32)
                        history[split_sample]['best_corr_val'] = progress[split_sample]['corr_val']
                        history[split_sample]['best_corr_test'] = progress[split_sample]['corr_test']
                print(f"({split_sample}) corr_val: {np.mean(progress[split_sample][f'corr_val']):.3f}, corr_test: {np.mean(progress[split_sample][f'corr_test']):.3f}")
                # Record training progress
                if split_sample == 'train':
                    # record losses & discriminator probabilities
                    for domain in ['source', 'target']:
                        for loss_type in ['loss_adv', 'loss_cyc', 'loss_id', 'loss_div']:
                            loss = float(np.mean(record[split_sample][domain]['trans'][loss_type]))
                            history[split_sample][domain][loss_type].append(loss)
                        for real_fake in ['real', 'fake']:
                            disc_pred = float(np.concatenate(record[split_sample][domain]['disc'][f'pred_{real_fake}']).mean())
                            history[split_sample][domain][f'disc_pred_{real_fake}'].append(disc_pred)
        # save progress history
        json.dump(history, open(os.path.join(self.log_dir, 'progress_log', 'progress.json'), 'w'))
        # save weights
        torch.save(best_weight_enc_source, os.path.join(self.log_dir, 'weights', 'enc_source.pth'))
        torch.save(best_weight_dec_target, os.path.join(self.log_dir, 'weights', 'dec_target.pth'))
        torch.save(best_weight_trans_s2t, os.path.join(self.log_dir, 'weights', 'trans_s2t.pth'))
        # save predictions
        dict_predictions = {}
        for split_sample in best_pred.keys():
            dict_predictions[split_sample] = {}
            for split_gene in best_pred[split_sample].keys():
                if split_sample == 'test' and best_pred[split_sample][split_gene] is None:
                    df = best_pred['val'][split_gene]
                else:
                    df = best_pred[split_sample][split_gene]
                df.to_pickle(os.path.join(self.log_dir, 'predictions', f'best_pred_sample-{split_sample}_gene-{split_gene}.pkl'))
        
        return best_pred['test']['test']
        
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
    # Decoder for target
    decoder_target = Linear1DBlock(model_opt['latent_dim_target'], in_features_target, model_opt['dec_features_target'], False, False)
    # Discriminator for target
    disc_target = Linear1DBlock(model_opt['latent_dim_target'], 1, model_opt['disc_features_target'], False, False, 0.1)
    # Translator from target to source
    trans_t2s = Linear1DBlock(model_opt['latent_dim_target'], latent_dim_source, model_opt['trans_features_t2s'], True, True, 0.1)
    # Define trainer
    trainer = Trainer(encoder_source, decoder_source, trans_s2t, disc_source, encoder_target, decoder_target, trans_t2s, disc_target, opts, df_source)
    return trainer
