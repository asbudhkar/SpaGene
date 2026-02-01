from codes.data_utils import (
    load_dataframe, process_dataframe,
    ExpDataset, CompositeDataset,
    composite_collate_fn,
)
from codes.train_utils import define_trainer

import yaml, os, argparse, numpy as np, pandas as pd, functools, tifffile as tif, glob
import torch


"""
Training procedure
1. Load experiment settings
2. Load base dataframe/ image and collect gene expression names
3. For each fold in cross validation, do..
    1. Define dataset and loader
    2. Define model
    3. Train
        1. Train Encoder-Decoder
        2. Train Translator
4. Save progress
"""

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_opt', type=str, default='base', dest='exp',
                        help='yaml file name for experiment (without extension)')
    return parser


def make_loader(ds, batch_size, shuffle, drop_last, num_workers, seed=1234, collate_fn=None):
    def seed_worker(worker_id):
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        import random as _random
        _random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(int(seed))

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g if shuffle else None,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    return torch.utils.data.DataLoader(ds, **kwargs)


if __name__ == '__main__':
    import os, time, torch
    print("="*80)
    print("START TRAINING")
    print("PID:", os.getpid())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("cuda available:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device0 name:", torch.cuda.get_device_name(0))
    print("Time:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80)

    import time
    start_time = time.time()

    # Load experiment settings
    print('Loading experiment settings...')
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, '../beanfur/gene_exp/gene_exp_private/data/paired_datasets')

    args = get_args().parse_args()
    opts = yaml.safe_load(open(os.path.join(base_dir, 'options', 'base.yaml')))
    add_opts = yaml.safe_load(open(os.path.join(base_dir, 'options', f"{args.exp}.yaml")))
    for key1 in add_opts.keys():
        for key2 in add_opts[key1].keys():
            opts[key1][key2] = add_opts[key1][key2]
    opts['data_opt']['data_dir'] = data_dir

    print(opts)
    data_opt = opts['data_opt']
    model_opt = opts['model_opt']
    train_opt = opts['train_opt']

    import random

    def seed_everything(seed: int):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed = int(train_opt.get("seed", 0))
    seed_everything(seed)

    print(f"[repro] seed={seed}")

    print('Experiment settings loaded')

    # Load base dataframe and collect gene expression names
    print('Loading base dataframe and collect gene expression names')
    df_source = load_dataframe(data_opt['domain_source'], data_opt['data_dir'])
    df_target = load_dataframe(data_opt['domain_target'], data_opt['data_dir'])

    genes_to_keep = list(set(df_source.columns).intersection(set(df_target.columns)))

    df_source = process_dataframe(
        df_source,
        data_opt['min_count_gene_source'], data_opt['min_count_cell_source'],
        data_opt['min_density_gene_source'], data_opt['min_density_cell_source'],
        data_opt['gene_selection_count_source'], data_opt['clip_outlier_source'],
        data_opt['normalization_source'],
        genes_to_keep=genes_to_keep
    )
    df_target = process_dataframe(
        df_target,
        data_opt['min_count_gene_target'], data_opt['min_count_cell_target'],
        data_opt['min_density_gene_target'], data_opt['min_density_cell_target'],
        data_opt['gene_selection_count_target'], data_opt['clip_outlier_target'],
        data_opt['normalization_target'],
        genes_to_keep=genes_to_keep
    )

    source_input_type = '1d' if model_opt['enc_type_source'].split('_')[0] == '1d' else 'composite'

    list_gene_id_source = sorted(df_source.columns)
    list_gene_id_target = sorted(df_target.columns)
    df_source = df_source.loc[:, list_gene_id_source]
    df_target = df_target.loc[:, list_gene_id_target]

    gene_exp_union = sorted(set(list_gene_id_source).union(set(list_gene_id_target)))
    gene_exp_inter = sorted(set(list_gene_id_source).intersection(set(list_gene_id_target)))

    gene_exp_input_target = [c for c in df_target.columns]

    print('Processed base dataframe loaded')
    print(f'source dataframe shape: {df_source.shape}')
    print(f'target dataframe shape: {df_target.shape}')
    print(f'# of union gene expression: {len(gene_exp_union)}')
    print(f'# of intersecting gene expression: {len(gene_exp_inter)}')

    # CV folds
    cv_gene   = int(train_opt['cv_gene'])
    cv_sample = int(train_opt['cv_sample'])

    list_fold_gene   = [i for i in range(cv_gene)]   if train_opt.get('target_fold_gene')   is None else [int(train_opt['target_fold_gene'])]
    list_fold_sample = [i for i in range(cv_sample)] if train_opt.get('target_fold_sample') is None else [int(train_opt['target_fold_sample'])]

    os.makedirs(train_opt['log_dir'], exist_ok=True)

    fold_gene_path   = os.path.join(train_opt['log_dir'], "fold_gene_assignments.csv")
    fold_sample_path = os.path.join(train_opt['log_dir'], "fold_sample_assignments.csv")

    if os.path.exists(fold_gene_path):
        fg = pd.read_csv(fold_gene_path)
        gene2fold = dict(zip(fg["gene"], fg["fold"]))
    else:
        gene2fold = {g: (i % cv_gene) for i, g in enumerate(gene_exp_inter)}
        pd.DataFrame({"gene": list(gene2fold.keys()), "fold": list(gene2fold.values())}).to_csv(fold_gene_path, index=False)

    if os.path.exists(fold_sample_path):
        fs = pd.read_csv(fold_sample_path)
        cell2fold = dict(zip(fs["cell_id"], fs["fold"]))
    else:
        cell2fold = {cid: (i % cv_sample) for i, cid in enumerate(df_source.index.tolist())}
        pd.DataFrame({"cell_id": list(cell2fold.keys()), "fold": list(cell2fold.values())}).to_csv(fold_sample_path, index=False)

    opts['gene_names'] = {
        'source': df_source.columns.tolist(),
        'target': df_target.columns.tolist(),
        'intersection': gene_exp_inter,
        'fold': {fold: [g for g in gene_exp_inter if gene2fold[g] == fold] for fold in list_fold_gene}
    }
    opts['cell_ids'] = {fold: [cid for cid in df_source.index.tolist() if cell2fold[cid] == fold] for fold in list_fold_sample}

    yaml.safe_dump(opts, open(os.path.join(train_opt['log_dir'], 'exp_setting.yaml'), 'w'))

    # Run CV
    for fold_gene in list_fold_gene:
        for fold_sample in list_fold_sample:
            print(f'Running experiment on fold_gene: {fold_gene+1}/{len(list_fold_gene)} and fold_sample: {fold_sample+1}/{len(list_fold_sample)}')

            gene_exp_val  = [exp for idx, exp in enumerate(gene_exp_inter) if (idx % train_opt['cv_gene']) != fold_gene]
            gene_exp_test = [exp for idx, exp in enumerate(gene_exp_inter) if (idx % train_opt['cv_gene']) == fold_gene]
            gene_exp_input_source = [c for c in df_source.columns if c not in gene_exp_test]

            if train_opt['cv_sample'] == 1:
                list_train_cell_id_source = df_source.index.tolist()
                list_val_cell_id_source   = df_source.index.tolist()
                list_test_cell_id_source  = df_source.index.tolist()
            else:
                list_dev_cell_id_source = [cell_id for idx, cell_id in enumerate(df_source.index) if (idx % train_opt['cv_sample']) != fold_sample]
                list_train_cell_id_source = [cell_id for idx, cell_id in enumerate(list_dev_cell_id_source) if idx % 10 != 0]
                list_val_cell_id_source   = [cell_id for idx, cell_id in enumerate(list_dev_cell_id_source) if idx % 10 == 0]
                list_test_cell_id_source  = [cell_id for idx, cell_id in enumerate(df_source.index) if (idx % train_opt['cv_sample']) == fold_sample]

            opts['exp_setting'] = {
                'fold_gene': fold_gene,
                'fold_sample': fold_sample,
                'cv_gene': train_opt['cv_gene'],
                'cv_sample': train_opt['cv_sample'],
                'cell_id': {
                    'train': list_train_cell_id_source,
                    'val':   list_val_cell_id_source,
                    'test':  list_test_cell_id_source
                },
                'gene_names': {
                    'val': gene_exp_val,
                    'test': gene_exp_test,
                    'source': df_source.columns.tolist(),
                    'source_input': gene_exp_input_source,
                    'target': df_target.columns.tolist()
                }
            }

            if train_opt['cv_sample'] == 1:
                dict_df = {
                    'source': {
                        'train': df_source.loc[list_train_cell_id_source],
                        'val':   df_source.loc[list_val_cell_id_source]
                    },
                    'target': {'train': df_target},
                }
            else:
                dict_df = {
                    'source': {
                        'train': df_source.loc[list_train_cell_id_source],
                        'val':   df_source.loc[list_val_cell_id_source],
                        'test':  df_source.loc[list_test_cell_id_source]
                    },
                    'target': {'train': df_target},
                }

            # Datasets
            dict_ds = {'target': {'train': ExpDataset(df_target, gene_exp_input_target)}}

            if source_input_type == '1d':
                dict_ds['source'] = {
                    split: ExpDataset(dict_df['source'][split], gene_exp_input_source, gene_exp_test)
                    for split in dict_df['source'].keys()
                }
            else:
                if data_opt['domain_source'] == 'nanostring':
                    image_dir = os.path.join(data_dir, 'nanostring', 'image')
                    label_dir = os.path.join(data_dir, 'nanostring', 'image_label')
                    df_meta = pd.read_csv(os.path.join(data_dir, data_opt['domain_source'], 'Lung9_Rep1_metadata_file.csv'))

                    dict_image_data = {
                        'image': {
                            fov: tif.imread(glob.glob(os.path.join(image_dir, f'*F{str(fov).zfill(3)}_Z003*'))[0])
                            for fov in range(1, 21)
                        },
                        'label': {
                            fov: tif.imread(os.path.join(label_dir, f'CellLabels_F{str(fov).zfill(3)}.tif'))
                            for fov in range(1, 21)
                        }
                    }

                    dict_image_data['meta'] = {}
                    for index in df_source.index:
                        fov = int(index.split('-')[0].split('_')[-1])
                        cell_id = int(index.split('-')[1])
                        cx, cy, width, height = df_meta.loc[
                            (df_meta['fov'] == fov) & (df_meta['cell_ID'] == cell_id),
                            ['CenterX_local_px', 'CenterY_local_px', 'Width', 'Height']
                        ].values[0]
                        dict_image_data['meta'][index] = dict(cx=float(cx), cy=float(cy), width=float(width), height=float(height))

                sampled_indices = []
                sample_size = len(dict_df['source']['train'])
                while len(sampled_indices) < sample_size:
                    index = np.random.choice(dict_df['source']['train'].index)
                    if all(
                        np.linalg.norm(df_source.loc[index][['x', 'y']].values -
                                       df_source.loc[sampled_index][['x', 'y']].values) > 5
                        for sampled_index in sampled_indices
                    ):
                        sampled_indices.append(index)

                target_size = tuple(opts['data_opt']['input_size_image'])
                dict_ds['source'] = {
                    split: CompositeDataset(
                        dict_df['source'][split].loc[sampled_indices],
                        dict_image_data,
                        gene_exp_input_source,
                        gene_exp_test,
                        target_size=target_size
                    )
                    for split in dict_df['source'].keys()
                }

            # DataLoaders
            dict_dl = {}
            dict_dl['target'] = {
                'train': make_loader(
                    dict_ds['target']['train'],
                    batch_size=train_opt['batch_size'],
                    shuffle=True,
                    drop_last=True,
                    num_workers=train_opt['num_workers'],
                )
            }

            dict_dl['source'] = {}
            for split in dict_ds['source'].keys():
                shuffle = (split == 'train')
                drop_last = (split == 'train')

                if source_input_type == '1d':
                    dict_dl['source'][split] = make_loader(
                        dict_ds['source'][split],
                        batch_size=train_opt['batch_size'],
                        shuffle=shuffle,
                        drop_last=drop_last,
                        num_workers=train_opt['num_workers'],
                    )
                else:
                    dict_dl['source'][split] = make_loader(
                        dict_ds['source'][split],
                        batch_size=train_opt['batch_size'],
                        shuffle=shuffle,
                        drop_last=drop_last,
                        num_workers=train_opt['num_workers'],
                        collate_fn=composite_collate_fn
                    )

            # Train
            trainer = define_trainer(len(gene_exp_input_source), len(gene_exp_input_target), opts, df_source)
            trainer.train_enc_dec(dict_dl)
            trainer.train(dict_dl)

    # Save prediction results and performance
    print('Saving prediction results and performance...')
    result_dir = train_opt['log_dir']
    os.makedirs(result_dir, exist_ok=True)

    pd.DataFrame(gene_exp_inter, columns=['Common gene']).to_csv('common_genes.csv', index=False)

    exp_global = yaml.safe_load(open(os.path.join(result_dir, 'exp_setting.yaml')))
    cv_gene = exp_global['train_opt']['cv_gene']
    cv_sample = exp_global['train_opt']['cv_sample']
    list_fold_gene = [i for i in range(cv_gene)] if exp_global['train_opt']['target_fold_gene'] is None else [exp_global['train_opt']['target_fold_gene']]
    list_fold_sample = [i for i in range(cv_sample)] if exp_global['train_opt']['target_fold_sample'] is None else [exp_global['train_opt']['target_fold_sample']]

    df_pred_parts = []
    gene_fold = {}

    for fold_gene in list_fold_gene:
        list_df = []
        for fold_sample in list_fold_sample:
            df_temp = pd.read_pickle(
                os.path.join(
                    result_dir,
                    f'fold_gene_{fold_gene}/fold_sample_{fold_sample}/predictions/best_pred_sample-test_gene-test.pkl'
                )
            )

            cell_ids   = exp_global['cell_ids'][fold_sample]
            gene_names = exp_global['gene_names']['fold'][fold_gene]
            list_df.append(df_temp.reindex(index=cell_ids, columns=gene_names))

        df_fold = pd.concat(list_df)
        df_pred_parts.append(df_fold)
        gene_fold.update({c: fold_gene for c in df_fold.columns})

    df_pred = pd.concat(df_pred_parts, axis=1)

    list_gene_exp = gene_exp_inter
    list_cell_id = df_pred.index

    df_real = df_source.reindex(index=list_cell_id, columns=list_gene_exp)
    df_pred = df_pred.reindex(index=list_cell_id, columns=list_gene_exp)

    df_real.to_csv(os.path.join(result_dir, 'real.csv'))
    df_pred.to_csv(os.path.join(result_dir, 'pred.csv'))

    # Metric helpers
    def pearson_vec(r, p):
        r = np.asarray(r, float)
        p = np.asarray(p, float)
        if np.std(r) == 0 or np.std(p) == 0:
            return np.nan
        return float(np.corrcoef(r, p)[0, 1])

    def rmse_vec(r, p):
        r = np.asarray(r, float)
        p = np.asarray(p, float)
        r = (r - np.mean(r)) / (np.std(r) + 1e-8)
        p = (p - np.mean(p)) / (np.std(p) + 1e-8)
        return float(np.sqrt(((r - p) ** 2).mean()))

    def compute_ssim_vec(x, y, C1=0.01, C2=0.03):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
        y_scaled = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
        ux, uy = np.mean(x_scaled), np.mean(y_scaled)
        var_x, var_y = np.var(x_scaled), np.var(y_scaled)
        cov_xy = np.cov(x_scaled, y_scaled)[0, 1]
        num = (2 * ux * uy + C1) * (2 * cov_xy + C2)
        den = (ux**2 + uy**2 + C1) * (var_x + var_y + C2)
        return float(num / den)

    def masked_metrics(r, p, mask, min_n=10):
        mask = np.asarray(mask, bool)
        n_used = int(mask.sum())
        if n_used < min_n:
            return np.nan, np.nan, np.nan, n_used
        rr = np.asarray(r, float)[mask]
        pp = np.asarray(p, float)[mask]
        cc = pearson_vec(rr, pp)
        if not np.isfinite(cc):
            return np.nan, np.nan, np.nan, n_used
        rm = rmse_vec(rr, pp)
        ss = compute_ssim_vec(rr, pp)
        if not (np.isfinite(rm) and np.isfinite(ss)):
            return np.nan, np.nan, np.nan, n_used
        return cc, rm, ss, n_used

    eps = 0.0
    pred_thr = 0.0
    min_n = 10

    list_corr, list_rmse, list_ssim, list_fold = [], [], [], []
    for gene_name in list_gene_exp:
        r = df_real.loc[:, gene_name].values
        p = df_pred.loc[:, gene_name].values
        cc = pearson_vec(r, p)
        list_corr.append(float(cc) if np.isfinite(cc) else np.nan)
        list_rmse.append(rmse_vec(r, p))
        list_ssim.append(compute_ssim_vec(r, p))
        list_fold.append(gene_fold[gene_name])

    df_performance = pd.DataFrame({
        'gene_name': list_gene_exp,
        'corr': list_corr,
        'rmse': list_rmse,
        'ssim': list_ssim,
        'cv_fold': list_fold
    })
    df_performance.to_csv(os.path.join(result_dir, 'performance.csv'), index=False)

    df_per_fold_mean = df_performance.groupby('cv_fold').mean(numeric_only=True)
    final_average = df_per_fold_mean.mean(numeric_only=True)
    pd.DataFrame(final_average).reset_index().rename(
        columns={0: "Final Average Value", "index": "Metric"}
    ).to_csv(os.path.join(result_dir, 'final_average_values.csv'), index=False)

    global_avg = df_performance[['corr', 'rmse', 'ssim']].mean(numeric_only=True)
    pd.DataFrame(global_avg).reset_index().rename(
        columns={0: "Global Average Value", "index": "Metric"}
    ).to_csv(os.path.join(result_dir, 'global_average_values.csv'), index=False)

    print(f'Prediction and performance per gene saved in {result_dir}')

    # Full data training and imputation
    print('\n========== TRAINING FINAL IMPUTER ON ALL DATA ==========')
    gene_exp_input_source = df_source.columns.tolist()
    gene_exp_input_target = df_target.columns.tolist()

    dict_ds_full = {
        'source': {'train': ExpDataset(df_source, gene_exp_input_source)},
        'target': {'train': ExpDataset(df_target, gene_exp_input_target)}
    }

    dict_dl_full = {
        'source': {
            'train': make_loader(dict_ds_full['source']['train'], train_opt['batch_size'], True,  True,  train_opt['num_workers']),
            'val':   make_loader(dict_ds_full['source']['train'], train_opt['batch_size'], False, True,  train_opt['num_workers']),
        },
        'target': {
            'train': make_loader(dict_ds_full['target']['train'], train_opt['batch_size'], True,  True,  train_opt['num_workers']),
            'val':   make_loader(dict_ds_full['target']['train'], train_opt['batch_size'], False, True,  train_opt['num_workers']),
        }
    }

    trainer_full = define_trainer(len(gene_exp_input_source), len(gene_exp_input_target), opts, df_source)

    print('[Full] Training encoder-decoder...')
    trainer_full.train_enc_dec(dict_dl_full)
    print('[Full] Training translator...')
    trainer_full.train(dict_dl_full)

    print('[Full] Predicting imputed matrix for all cells & genes...')
    trainer_full.enc_source.eval()
    trainer_full.trans_s2t.eval()
    trainer_full.dec_target.eval()

    use_amp = bool(train_opt.get("use_amp", True))
    device = trainer_full.device

    all_cell_ids = []
    all_preds = []
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=use_amp):
        for batch in dict_dl_full['source']['train']:
            x = batch['input'].to(device, non_blocking=True)
            pred = trainer_full.translate_s2t({'input': x})
            all_preds.append(pred.float().cpu().numpy())
            all_cell_ids.extend(batch['index'])

    pred_matrix = np.concatenate(all_preds, axis=0)
    df_full_pred = pd.DataFrame(pred_matrix, index=all_cell_ids, columns=gene_exp_input_target)

    out_path = os.path.join(train_opt['log_dir'], 'full_data_imputed.csv')
    df_full_pred.to_csv(out_path)
    print(f'[Full] Full-data imputed matrix saved to {out_path}')

    end_time = time.time()
    elapsed_sec = end_time - start_time
    print(f"Total training time: {elapsed_sec:.2f} seconds ({elapsed_sec/60:.2f} min, {elapsed_sec/3600:.2f} hr)")
    import time
    print("="*80)
    print("END TRAINING (SUCCESS)")
    print("Time:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80)
