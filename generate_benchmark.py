from codes.data_utils import load_dataframe, process_dataframe
import yaml, os, argparse, json, numpy as np, pandas as pd, scanpy as sc
from skimage.metrics import structural_similarity as ssim
import scipy.stats as st
import sys
from scvi.external import GIMVI
sys.path.append(os.path.join(os.path.dirname(__file__), 'SpaGE/'))
from SpaGE.main import SpaGE

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_opt', type=str, default='base', dest='exp',
                        help='yaml file name for experiment (without extension)')
    return parser

def rmse(r, p):
    r = (r - np.mean(r)) / np.std(r)
    p = (p - np.mean(p)) / np.std(p)
    return np.sqrt(((r - p) ** 2).mean())
    
# Function to compute SSIM
def compute_ssim(x, y, C1=0.01, C2=0.03):
    # Scale x and y between 0 and 1
    x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
    y_scaled = (y - np.min(y)) / (np.max(y) - np.min(y))
    
    # Compute means, variances, and covariance
    ux, uy = np.mean(x_scaled), np.mean(y_scaled)
    var_x, var_y = np.var(x_scaled), np.var(y_scaled)
    cov_xy = np.cov(x_scaled, y_scaled)[0, 1]
    
    # SSIM calculation based on the formula in the image
    numerator = (2 * ux * uy + C1) * (2 * cov_xy + C2)
    denominator = (ux**2 + uy**2 + C1) * (var_x + var_y + C2)
    
    return numerator / denominator

def scale_plus(arr):
    return arr / np.sum(arr)


if __name__ == '__main__':
    """
    1. Load experiment settings
    """
    print('Loading experiment settings...')
    # Define path settings
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, '../beanfur/gene_exp/gene_exp_private/data/paired_datasets')
    
    # Define user-defined experiment settings
    parser = get_args()
    args = parser.parse_args()
    opts = yaml.safe_load(open(os.path.join(base_dir, 'options', 'base.yaml')))
    add_opts = yaml.safe_load(open(os.path.join(base_dir, 'options', f"{args.exp}.yaml")))
    for key1 in add_opts.keys():
        for key2 in add_opts[key1].keys():
            opts[key1][key2] = add_opts[key1][key2]
    opts['data_opt']['data_dir'] = data_dir
    data_opt = opts['data_opt']
    model_opt = opts['model_opt']
    train_opt = opts['train_opt']
    print('Experiment settings loaded')
    print(opts)
    """
    2. Load base dataframe and collect gene expression names
    """
    print('Loading base dataframe and collect gene expression names')
    # Load dataframe
    df_source = load_dataframe(data_opt['domain_source'], data_opt['data_dir'])
    df_target = load_dataframe(data_opt['domain_target'], data_opt['data_dir'])


    genes_to_keep = list(set(df_source.columns).intersection(set(df_target.columns)))

    # Process dataframe
    df_source = process_dataframe(df_source, data_opt['min_count_gene_source'], data_opt['min_count_cell_source'], data_opt['min_density_gene_source'], data_opt['min_density_cell_source'], data_opt['gene_selection_count_source'], data_opt['clip_outlier_source'], data_opt['normalization_source'], genes_to_keep = genes_to_keep)
    df_target = process_dataframe(df_target, data_opt['min_count_gene_target'], data_opt['min_count_cell_target'], data_opt['min_density_gene_target'], data_opt['min_density_cell_target'], data_opt['gene_selection_count_target'], data_opt['clip_outlier_target'], data_opt['normalization_target'], genes_to_keep = genes_to_keep)
    

    # Set index and sort dataframe by names - just for convenience
    list_cell_id_source = df_source.index
    list_cell_id_target = df_target.index
    list_gene_id_source = sorted(df_source.columns)
    list_gene_id_target = sorted(df_target.columns)
    df_source = df_source.loc[:, list_gene_id_source]
    df_target = df_target.loc[:, list_gene_id_target]

    # Find common genes by set intersection
    gene_exp_union = sorted(set(list_gene_id_source).union(set(list_gene_id_target)))
    gene_exp_inter = sorted(set(list_gene_id_source).intersection(set(list_gene_id_target)))
    # Set input gene expression
    gene_exp_input_target = [c for c in df_target.columns]
    print('Base dataframe loaded')
    print(f'Source dataframe shape: {df_source.shape}')
    print(f'Target dataframe shape: {df_target.shape}')
    print(f'# of union gene expression: {len(gene_exp_union)}')
    print(f'# of intersecting gene expression: {len(gene_exp_inter)}')
    """
    3. For each fold in cross validation, do..
        1. Define dataset and loader
        2. Define model
        3. Train
            1. Train Encoder-Decoder
            2. Train Translator
    """
    # Define list of folds to run
    list_fold_gene = [i for i in range(train_opt['cv_gene'])] if train_opt['target_fold_gene'] is None else [train_opt['target_fold_gene']]
    list_fold_sample = [i for i in range(train_opt['cv_sample'])] if train_opt['target_fold_sample'] is None else [train_opt['target_fold_sample']]
    # Save opts
    os.makedirs(train_opt['log_dir'], exist_ok=True)
    yaml.safe_dump(opts, open(os.path.join(train_opt['log_dir'], 'exp_setting.yaml'), 'w'))
    
    # Initialize metrics list for each fold
    all_fold_metrics = []
    all_fold_metrics1 = []
    # For each fold in cross validation...
    for fold_gene in list_fold_gene:
        for fold_sample in list_fold_sample:
            print(f'Running experiment on fold_gene: {fold_gene+1}/{len(list_fold_gene)} and fold_sample: {fold_sample+1}/{len(list_fold_sample)}')
            """
            3-1. Define dataset and loader
            """
            # Set validation and test gene expressions
            gene_exp_val = [exp for idx, exp in enumerate(gene_exp_inter) if (idx % train_opt['cv_gene']) != fold_gene]
            gene_exp_test = [exp for idx, exp in enumerate(gene_exp_inter) if (idx % train_opt['cv_gene']) == fold_gene]
            # Set input gene expressions for source domain
            gene_exp_input_source = [c for c in df_source.columns if c not in gene_exp_test]
            if train_opt['cv_sample'] == 1:
                list_train_cell_id_source = df_source.index.tolist()
                list_val_cell_id_source = df_source.index.tolist()
                list_test_cell_id_source = df_source.index.tolist()
            else:
                # Split dataset for source (target is only used for referencing)
                list_dev_cell_id_source = [cell_id for idx, cell_id in enumerate(df_source.index) if (idx % train_opt['cv_sample']) != fold_sample]
                list_train_cell_id_source = [cell_id for idx, cell_id in enumerate(list_dev_cell_id_source) if idx % 10 != 0]
                list_val_cell_id_source = [cell_id for idx, cell_id in enumerate(list_dev_cell_id_source) if idx % 10 == 0]
                list_test_cell_id_source = [cell_id for idx, cell_id in enumerate(df_source.index) if (idx % train_opt['cv_sample']) == fold_sample]
            dict_df = {
                'source': {
                    'train': df_source.loc[list_train_cell_id_source],
                    'val': df_source.loc[list_val_cell_id_source],
                    'test': df_source.loc[list_test_cell_id_source]
                },
                'target': {
                    'train': df_target
                }
            }
            
            # SPAGE
            if model_opt['method'] == 'spage':
                print('Running SpaGE')
                ImpGene = SpaGE(dict_df['source']['train'].drop(columns=gene_exp_test), dict_df['target']['train'], n_pv=10, genes_to_predict=gene_exp_test)
                pred = ImpGene.loc[:, gene_exp_test]
                real = dict_df['source']['train'].loc[:, gene_exp_test]
            
            # GIMVI
            if model_opt['method'] == 'gimvi':
                print('Running GIMVI')
                source_data = sc.AnnData(dict_df['source']['train'].loc[:, gene_exp_inter].drop(columns=gene_exp_test))
                target_data = sc.AnnData(dict_df['target']['train'])
                GIMVI.setup_anndata(target_data)
                GIMVI.setup_anndata(source_data)
                model = GIMVI(target_data, source_data)
                model.train(100)
                _, imputation = model.get_imputed_values(normalized=False)
                list_index_gene = [np.argwhere(np.array(dict_df['target']['train'].columns) == i)[0][0] for i in gene_exp_test]
                pred = pd.DataFrame(columns=dict_df['target']['train'].columns, index=dict_df['source']['train'].index, data=imputation).loc[:, gene_exp_test]
                real = dict_df['source']['train'].loc[:, gene_exp_test]
            
            # Save progress
            os.makedirs(os.path.join(train_opt['log_dir'], f'fold_gene_{fold_gene}', f'fold_sample_{fold_sample}'), exist_ok=True)
            pred.to_csv(os.path.join(train_opt['log_dir'], f'fold_gene_{fold_gene}', f'fold_sample_{fold_sample}', 'pred.csv'))
            real.to_csv(os.path.join(train_opt['log_dir'], f'fold_gene_{fold_gene}', f'fold_sample_{fold_sample}', 'real.csv'))

            # Initialize metrics for this fold
            list_corr = []
            list_ssim = []
            list_rmse = []
            list_js = []
            list_gene = []

            # Calculate metrics
            for col in pred.columns:
                r = real.loc[:, col].values
                p = pred.loc[:, col].values
                list_corr.append(np.corrcoef(r, p)[0, 1])
                list_gene.append(col)

                # Calculate and append RMSE
                list_rmse.append(rmse(r, p))

                # Calculate and append SSIM
                list_ssim.append(compute_ssim(r, p))


            all_fold_metrics.append({
                'ssim': list_ssim,
                'rmse': list_rmse,
                'corr': list_corr
            })
            
            for i, gene in enumerate(list_gene):
                all_fold_metrics1.append({
                    'gene_name': gene,
                    'fold': fold_gene,
                    'ssim': list_ssim[i],
                    'rmse': list_rmse[i],
                    'corr': list_corr[i],
                })

    # Convert metrics to DataFrame
    df_metrics = pd.DataFrame(all_fold_metrics)

    # Compute average and median metrics across all folds
    avg_metrics = df_metrics.apply(lambda x: pd.Series({
        'average_ssim': np.mean([np.mean(metrics) for metrics in x['ssim']]),
        'average_rmse': np.mean([np.mean(metrics) for metrics in x['rmse']]),
        'average_corr': np.mean([np.mean(metrics) for metrics in x['corr']]),
    }), axis=1)

    # Aggregate metrics across all folds
    aggregated_metrics = {
        'ssim_mean': np.mean([metric for fold_metrics in all_fold_metrics for metric in fold_metrics['ssim']]),
        'rmse_mean': np.mean([metric for fold_metrics in all_fold_metrics for metric in fold_metrics['rmse']]),
        'corr_mean': np.mean([metric for fold_metrics in all_fold_metrics for metric in fold_metrics['corr']]),
    }

    # Save metrics summary
    df_metrics_summary = pd.DataFrame(aggregated_metrics, index=[0])
    df_metrics_summary.to_csv(os.path.join(train_opt['log_dir'], 'metrics_summary.csv'), index=False)

    df_performance = pd.DataFrame(all_fold_metrics1)
    df_performance.to_csv(os.path.join(train_opt['log_dir'], 'performance.csv'), index=False)

    df_per_fold_mean = df_performance.groupby('fold').mean(numeric_only=True)
    final_average = df_per_fold_mean.mean()
    final_average.to_csv(os.path.join(train_opt['log_dir'], 'final_average_values.csv'), index=False)

    # Save final averages
    global_avg = df_performance[['ssim', 'rmse', 'corr']].mean()
    global_avg_df = pd.DataFrame(global_avg).reset_index()
    global_avg_df.columns = ['Metric', 'Global Average Value']
    global_avg_df.to_csv(os.path.join(train_opt['log_dir'], 'global_average_values.csv'), index=False)