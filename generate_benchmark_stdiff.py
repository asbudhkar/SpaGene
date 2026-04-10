from codes.data_utils import load_dataframe, process_dataframe
from codes.benchmark_utils import run_stdiff
import yaml, os, argparse, numpy as np, pandas as pd


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_opt', type=str, default='base', dest='exp',
                        help='yaml file name for experiment (without extension)')
    return parser

# Function to compute RMSE
def rmse(r, p):
    r = (r - np.mean(r)) / np.std(r)
    p = (p - np.mean(p)) / np.std(p)
    return np.sqrt(((r - p) ** 2).mean())

# Function to compute SSIM
def compute_ssim(x, y, C1=0.01, C2=0.03):
    x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
    y_scaled = (y - np.min(y)) / (np.max(y) - np.min(y))
    ux, uy = np.mean(x_scaled), np.mean(y_scaled)
    var_x, var_y = np.var(x_scaled), np.var(y_scaled)
    cov_xy = np.cov(x_scaled, y_scaled)[0, 1]
    numerator = (2 * ux * uy + C1) * (2 * cov_xy + C2)
    denominator = (ux**2 + uy**2 + C1) * (var_x + var_y + C2)
    return numerator / denominator

if __name__ == '__main__':
    import time
    start_time = time.time()
    """
    1. Load experiment settings
    """
    print('Loading experiment settings...')
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data/paired_datasets')

    args = get_args().parse_args()

    opts = yaml.safe_load(open(os.path.join(base_dir, 'options', 'base.yaml')))
    add_opts = yaml.safe_load(open(os.path.join(base_dir, 'options', f"{args.exp}.yaml")))
    for key1 in add_opts.keys():
        for key2 in add_opts[key1].keys():
            opts[key1][key2] = add_opts[key1][key2]

    opts['data_opt']['data_dir'] = data_dir
    data_opt = opts['data_opt']
    train_opt = opts['train_opt']
    model_opt = opts['model_opt']

    if str(model_opt.get('method', '')).lower() != 'stdiff':
        raise ValueError(f"generate_benchmark_stdiff.py expects method='stdiff', got {model_opt.get('method')}")

    print('Experiment settings loaded')
    print(opts)

    """
    2. Load base dataframe and collect gene expression names
    """
    print('Loading base dataframe and collect gene expression names')
    df_source_raw = load_dataframe(data_opt['domain_source'], data_opt['data_dir'])
    df_target_raw = load_dataframe(data_opt['domain_target'], data_opt['data_dir'])
    genes_to_keep = list(set(df_source_raw.columns).intersection(set(df_target_raw.columns)))

    df_source_full = process_dataframe(
        df_source_raw.copy(),
        data_opt['min_count_gene_source'], data_opt['min_count_cell_source'],
        data_opt['min_density_gene_source'], data_opt['min_density_cell_source'],
        data_opt['gene_selection_count_source'], data_opt['clip_outlier_source'],
        data_opt['normalization_source'],
        genes_to_keep=None
    )
    df_source = process_dataframe(
        df_source_raw,
        data_opt['min_count_gene_source'], data_opt['min_count_cell_source'],
        data_opt['min_density_gene_source'], data_opt['min_density_cell_source'],
        data_opt['gene_selection_count_source'], data_opt['clip_outlier_source'],
        data_opt['normalization_source'],
        genes_to_keep=genes_to_keep
    )
    df_target = process_dataframe(
        df_target_raw,
        data_opt['min_count_gene_target'], data_opt['min_count_cell_target'],
        data_opt['min_density_gene_target'], data_opt['min_density_cell_target'],
        data_opt['gene_selection_count_target'], data_opt['clip_outlier_target'],
        data_opt['normalization_target'],
        genes_to_keep=genes_to_keep
    )

    df_source_full = df_source_full.loc[:, sorted(df_source_full.columns)]
    df_source = df_source.loc[:, sorted(df_source.columns)]
    df_target = df_target.loc[:, sorted(df_target.columns)]

    gene_exp_inter = sorted(set(df_source.columns).intersection(set(df_target.columns)))
    gene_exp_union = sorted(set(df_source.columns).union(set(df_target.columns)))

    print('Base dataframe loaded')
    print(f'Source dataframe shape: {df_source.shape}')
    print(f'Target dataframe shape: {df_target.shape}')
    print(f'# of union gene expression: {len(gene_exp_union)}')
    print(f'# of intersecting gene expression: {len(gene_exp_inter)}')
    
    """
    3. CV folds
    """
    list_fold_gene = [i for i in range(train_opt['cv_gene'])] if train_opt['target_fold_gene'] is None else [train_opt['target_fold_gene']]
    list_fold_sample = [i for i in range(train_opt['cv_sample'])] if train_opt['target_fold_sample'] is None else [train_opt['target_fold_sample']]

    os.makedirs(train_opt['log_dir'], exist_ok=True)
    yaml.safe_dump(opts, open(os.path.join(train_opt['log_dir'], 'exp_setting.yaml'), 'w'))

    all_fold_metrics = []
    all_fold_metrics1 = []

    stdiff_use_all_st_genes = bool(train_opt.get('stdiff_use_all_st_genes', False))
    if stdiff_use_all_st_genes:
        df_source_model = df_source_full.copy()
        df_target_model = df_target.reindex(columns=df_source_model.columns, fill_value=0.0)
    else:
        df_source_model = df_source.loc[:, gene_exp_inter].copy()
        df_target_model = df_target.loc[:, gene_exp_inter].copy()

    # stDiff
    for fold_sample in list_fold_sample:
        print(f'Running stDiff for fold_sample: {fold_sample+1}/{len(list_fold_sample)}')

        if train_opt['cv_sample'] == 1:
            list_test_cell_id_source = df_source_model.index.tolist()
        else:
            list_test_cell_id_source = [
                cell_id for idx, cell_id in enumerate(df_source_model.index)
                if (idx % train_opt['cv_sample']) == fold_sample
            ]

        df_source_eval = df_source_model.loc[list_test_cell_id_source].copy()
        full_pred = run_stdiff(df_source_eval, df_target_model, train_opt)
        full_pred = full_pred.loc[df_source_eval.index.astype(str)]

        for fold_gene in list_fold_gene:
            print(f'Writing outputs for fold_gene: {fold_gene+1}/{len(list_fold_gene)} and fold_sample: {fold_sample+1}/{len(list_fold_sample)}')
            gene_exp_test = [exp for idx, exp in enumerate(gene_exp_inter) if (idx % train_opt['cv_gene']) == fold_gene]

            pred = full_pred.loc[:, gene_exp_test]
            real = df_source_eval.copy()
            real.index = real.index.astype(str)
            real = real.loc[pred.index, gene_exp_test]

            fold_dir = os.path.join(train_opt['log_dir'], f'fold_gene_{fold_gene}', f'fold_sample_{fold_sample}')
            os.makedirs(fold_dir, exist_ok=True)
            pred.to_csv(os.path.join(fold_dir, 'pred.csv'))
            real.to_csv(os.path.join(fold_dir, 'real.csv'))

            # Calculate metrics
            list_corr = []
            list_ssim = []
            list_rmse = []
            list_gene = []

            for col in pred.columns:
                r = real.loc[:, col].values
                p = pred.loc[:, col].values

                list_corr.append(np.corrcoef(r, p)[0, 1])
                list_gene.append(col)
                list_rmse.append(rmse(r, p))
                list_ssim.append(compute_ssim(r, p))

            all_fold_metrics.append({'ssim': list_ssim, 'rmse': list_rmse, 'corr': list_corr})
            for i, gene in enumerate(list_gene):
                all_fold_metrics1.append({
                    'gene_name': gene,
                    'fold': fold_gene,
                    'fold_sample': fold_sample,
                    'ssim': list_ssim[i],
                    'rmse': list_rmse[i],
                    'corr': list_corr[i],
                })

    # Save metrics summary
    aggregated_metrics = {
        'ssim_mean': np.mean([metric for fold_metrics in all_fold_metrics for metric in fold_metrics['ssim']]),
        'rmse_mean': np.mean([metric for fold_metrics in all_fold_metrics for metric in fold_metrics['rmse']]),
        'corr_mean': np.mean([metric for fold_metrics in all_fold_metrics for metric in fold_metrics['corr']]),
    }
    pd.DataFrame(aggregated_metrics, index=[0]).to_csv(os.path.join(train_opt['log_dir'], 'metrics_summary.csv'), index=False)

    df_performance = pd.DataFrame(all_fold_metrics1)
    df_performance.to_csv(os.path.join(train_opt['log_dir'], 'performance.csv'), index=False)

    df_per_fold_mean = df_performance.groupby('fold').mean(numeric_only=True)
    final_average = df_per_fold_mean.mean()
    final_average.to_csv(os.path.join(train_opt['log_dir'], 'final_average_values.csv'), index=False)

    global_avg = df_performance[['ssim', 'rmse', 'corr']].mean()
    global_avg_df = pd.DataFrame(global_avg).reset_index()
    global_avg_df.columns = ['Metric', 'Global Average Value']
    global_avg_df.to_csv(os.path.join(train_opt['log_dir'], 'global_average_values.csv'), index=False)

    elapsed_sec = time.time() - start_time
    print(f"Total training time: {elapsed_sec:.2f} seconds ({elapsed_sec/60:.2f} min, {elapsed_sec/3600:.2f} hr)")
