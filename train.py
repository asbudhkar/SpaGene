from codes.data_utils import load_dataframe, process_dataframe, ExpDataset, CompositeDataset, composite_collate_fn, normalize_minmax_numpy
from codes.train_utils import Trainer, define_loss, define_trainer

import yaml, os, argparse, json, numpy as np, pandas as pd, functools, tifffile as tif, glob
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
    parser.add_argument('--exp_opt', type = str, default = 'base', dest = 'exp',
                        help = 'yaml file name for experiment (without extension)')
    return parser

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
    print(opts)
    data_opt = opts['data_opt']
    model_opt = opts['model_opt']
    train_opt = opts['train_opt']
    print('Experiment settings loaded')

    """
    2. Load base dataframe and collect gene expression names
    """
    print('Loading base dataframe and collect gene expression names')
    # Load dataframe
    df_source = load_dataframe(data_opt['domain_source'], data_opt['data_dir'])
    df_target = load_dataframe(data_opt['domain_target'], data_opt['data_dir'])
    # Process datafame
    genes_to_keep = list(set(df_source.columns).intersection(set(df_target.columns)))

    df_source = process_dataframe(df_source, data_opt['min_count_gene_source'], data_opt['min_count_cell_source'], data_opt['min_density_gene_source'], data_opt['min_density_cell_source'], data_opt['gene_selection_count_source'], data_opt['clip_outlier_source'], data_opt['normalization_source'], genes_to_keep = genes_to_keep)
    df_target = process_dataframe(df_target, data_opt['min_count_gene_target'], data_opt['min_count_cell_target'], data_opt['min_density_gene_target'], data_opt['min_density_cell_target'], data_opt['gene_selection_count_target'], data_opt['clip_outlier_target'], data_opt['normalization_target'], genes_to_keep = genes_to_keep)

    # Load image (if needed)
    source_input_type = '1d' if model_opt['enc_type_source'].split('_')[0] == '1d' else 'composite'
    # Set index and sort dataframe by names - just for convenience
    list_cell_id_source = df_source.index
    list_cell_id_target = df_target.index
    list_gene_id_source = sorted(df_source.columns)
    list_gene_id_target = sorted(df_target.columns)
    df_source = df_source.loc[:,list_gene_id_source]
    df_target = df_target.loc[:,list_gene_id_target]
    
    # Find common genes by set intersection
    gene_exp_union = sorted(set(list_gene_id_source).union(set(list_gene_id_target)))
    gene_exp_inter = sorted(set(list_gene_id_source).intersection(set(list_gene_id_target)))
   
    # Set input gene expression
    gene_exp_input_target = [c for c in df_target.columns]
    print('Processed base dataframe loaded')
    print(f'source dataframe shape: {df_source.shape}')
    print(f'target dataframe shape: {df_target.shape}')
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
    os.makedirs(train_opt['log_dir'], exist_ok = True)
    opts['gene_names'] = {
        'source': df_source.columns.tolist(),
        'target': df_target.columns.tolist(),
        'intersection': gene_exp_inter,
        'fold': {
            fold: [exp for idx, exp in enumerate(gene_exp_inter) if (idx % train_opt['cv_gene']) == fold]
            for fold in list_fold_gene
        }
    }
    opts['cell_ids'] = {
        fold: [cell_id for idx, cell_id in enumerate(df_source.index) if (idx % train_opt['cv_sample']) == fold]
        for fold in list_fold_sample
    }
    yaml.safe_dump(opts, open(os.path.join(train_opt['log_dir'], 'exp_setting.yaml'), 'w'))
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
            # Define trainer
            # update setting
            opts['exp_setting'] = {
                'fold_gene': fold_gene,
                'fold_sample': fold_sample,
                'cv_gene': train_opt['cv_gene'],
                'cv_sample': train_opt['cv_sample'],
                'cell_id': {
                    'train': list_train_cell_id_source,
                    'val': list_val_cell_id_source,
                    'test': list_test_cell_id_source
                },
                'gene_names':{
                    'val': gene_exp_val,
                    'test': gene_exp_test,
                    'source': df_source.columns.tolist(),
                    'source_input': gene_exp_input_source,
                    'target': df_target.columns.tolist()
                }
            }
            # Split
            if train_opt['cv_sample'] == 1:
                dict_df = {
                    'source':{
                        'train': df_source.loc[list_train_cell_id_source],
                        'val': df_source.loc[list_val_cell_id_source]
                    },
                    'target': {
                        'train': df_target
                    },
                }
            else:
                dict_df = {
                    'source':{
                        'train': df_source.loc[list_train_cell_id_source],
                        'val': df_source.loc[list_val_cell_id_source],
                        'test': df_source.loc[list_test_cell_id_source]
                    },
                    'target': {
                        'train': df_target
                    },
                }

            # Initialize the datasets
            dict_ds = {
                'target': {
                    'train': ExpDataset(df_target, gene_exp_input_target)
                }
            }

            if source_input_type == '1d':
                dict_ds['source'] = {
                    split: ExpDataset(dict_df['source'][split], gene_exp_input_source, gene_exp_test)
                    for split in dict_df['source'].keys()
                }
                print(dict_ds['source'])
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
                        cx, cy, width, height = df_meta.loc[(df_meta['fov'] == fov) & (df_meta['cell_ID'] == cell_id), ['CenterX_local_px', 'CenterY_local_px', 'Width', 'Height']].values[0]
                        dict_image_data['meta'][index] = {
                            'cx': cx,
                            'cy': cy,
                            'width': width,
                            'height': height
                        }

                # Sampling from the source dataset
                sampled_indices = []
                sample_size = len(dict_df['source']['train'])  # or another defined size
                while len(sampled_indices) < sample_size:
                    index = np.random.choice(dict_df['source']['train'].index)
                    if all(np.linalg.norm(df_source.loc[index][['x', 'y']].values - df_source.loc[sampled_index][['x', 'y']].values) > 5 for sampled_index in sampled_indices):
                        sampled_indices.append(index)

                dict_ds['source'] = {
                    split: CompositeDataset(dict_df['source'][split].loc[sampled_indices], dict_image_data, gene_exp_input_source, gene_exp_test)
                    for split in dict_df['source'].keys()
                }

            # Define DataLoader
            dict_dl = {}
            dict_dl['target'] = {
                'train': torch.utils.data.DataLoader(dict_ds['target']['train'], batch_size=train_opt['batch_size'], shuffle=True, drop_last=True, num_workers=train_opt['num_workers'])
            }

            domain = 'source'
            dict_dl[domain] = {}
            for split in dict_ds[domain].keys():
                shuffle = True if split == 'train' else False
                drop_last = True if split == 'train' else False
                if source_input_type == '1d':
                    dict_dl[domain][split] = torch.utils.data.DataLoader(dict_ds[domain][split], batch_size=train_opt['batch_size'], shuffle=shuffle, drop_last=drop_last, num_workers=train_opt['num_workers'])
                else:
                    target_size = opts['data_opt']['input_size_image']
                    exclude_background = opts['data_opt']['exclude_background']
                    normalize_method = opts['data_opt']['normalization_img']
                    dict_dl[domain][split] = torch.utils.data.DataLoader(dict_ds[domain][split], batch_size=train_opt['batch_size'], shuffle=shuffle, drop_last=drop_last, num_workers=train_opt['num_workers'], collate_fn=functools.partial(composite_collate_fn, target_size=target_size, exclude_background=exclude_background, normalize_method=normalize_method))

            """
            3-2. Define model
            """
            # Define trainer
            trainer = define_trainer(len(gene_exp_input_source), len(gene_exp_input_target), opts, df_source)
            """
            3-3. Train
            """
            # 3-3-1. Train Encoder Decoder
            trainer.train_enc_dec(dict_dl)
            # 3-3-2. Train translator
            trainer.train(dict_dl)
    """
    4. Save prediction results and performance
    """
    print('Saving prediction results and performance...')
    result_dir = train_opt['log_dir']

    df_correlation = pd.DataFrame(gene_exp_inter, columns=['Common gene'])
    df_correlation.to_csv('common_genes.csv', index=False)

    print("Correlation values saved to 'correlation_non_zero.csv'")
    exp_global = yaml.safe_load(open(os.path.join(result_dir, 'exp_setting.yaml')))
    cv_gene = exp_global['train_opt']['cv_gene']
    cv_sample = exp_global['train_opt']['cv_sample']
    list_fold_gene = [i for i in range(cv_gene)] if exp_global['train_opt']['target_fold_gene'] is None else [exp_global['train_opt']['target_fold_gene']]
    list_fold_sample = [i for i in range(cv_sample)] if exp_global['train_opt']['target_fold_sample'] is None else [exp_global['train_opt']['target_fold_sample']]
    df_pred = []
    gene_fold = {}
    for fold_gene in list_fold_gene:
        list_df = []
        for fold_sample in list_fold_sample:
            df_temp = pd.read_pickle(os.path.join(result_dir, f'fold_gene_{fold_gene}/fold_sample_{fold_sample}/predictions/best_pred_sample-test_gene-test.pkl'))
            cell_ids = opts['cell_ids'][fold_sample]
            gene_names = opts['gene_names']['fold'][fold_gene]
            list_df.append(df_temp.loc[cell_ids, gene_names])
        df = pd.concat(list_df)
        df_pred.append(df)
        gene_fold.update({c: fold_gene for c in df.columns})
    df_pred = pd.concat(df_pred, axis = 1)
    list_gene_exp = gene_exp_inter #df_pred.columns
    list_cell_id = df_pred.index
    df_real = df_source.loc[list_cell_id, list_gene_exp]
    df_pred = df_pred.loc[list_cell_id, list_gene_exp]
    df_real.to_csv(os.path.join(result_dir, 'real.csv'))
    df_pred.to_csv(os.path.join(result_dir, 'pred.csv'))
    
    # Record performance

    from skimage.metrics import structural_similarity as ssim
    from scipy.spatial.distance import jensenshannon
    # Function to compute RMSE
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
    
    import scipy.stats as st
    # Scale function for numpy arrays
    def scale_plus(arr):
        return arr / np.sum(arr)


    list_corr = []
    list_ssim = []
    list_rmse = []
    list_js = []
    list_fold = []
    list_density = []

    for gene_name in list_gene_exp:
        r = df_real.loc[:, gene_name].values
        p = df_pred.loc[:, gene_name].values
        
        # Calculate correlation
        list_corr.append(np.corrcoef(r, p)[0, 1])
        
        # Calculate and append RMSE
        rmse_value = rmse(r, p)
        list_rmse.append(rmse_value)

        # Calculate and append SSIM
        ssim_value = compute_ssim(r, p)
        list_ssim.append(ssim_value)

        # Append fold information
        list_fold.append(gene_fold[gene_name])

    # Create a DataFrame to store per-gene performance
    df_performance = pd.DataFrame({
        'gene_name': list_gene_exp,
        'corr': list_corr,
        'rmse': list_rmse,
        'ssim': list_ssim,
        'cv_fold': list_fold
    })

    # Save the per-gene performance metrics
    df_performance.to_csv(os.path.join(result_dir, 'performance.csv'))

    # Group by fold to calculate per-fold metrics (mean and median per fold)
    df_per_fold_mean = df_performance.groupby('cv_fold').mean(numeric_only=True)
    df_per_fold_median = df_performance.groupby('cv_fold').median(numeric_only=True)

    # Now calculate the final averages across all folds
    final_average = df_per_fold_mean.mean()

    # Convert the average Series to a DataFrame for easier saving
    final_average_df = pd.DataFrame(final_average).reset_index()
    final_average_df.columns = ['Metric', 'Final Average Value']

    # Save the final averages to CSV
    final_average_df.to_csv(os.path.join(result_dir, 'final_average_values.csv'), index=False)

    global_avg = df_performance[['corr', 'rmse', 'ssim']].mean()
    global_avg_df = pd.DataFrame(global_avg).reset_index()
    global_avg_df.columns = ['Metric', 'Global Average Value']
    global_avg_df.to_csv(os.path.join(result_dir, 'global_average_values.csv'), index=False)
