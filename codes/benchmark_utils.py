""
Most of codes are based on benchmarking paper
"""

import sys, os, time, pickle, numpy as np, pandas as pd
from .train_utils import define_trainer
from .data_utils import ExpDataset
import torch

def filter_sc(df, min_features = 200):
    """
    ref - https://github.com/satijalab/seurat/blob/86a817a4368d9f0cbf6689abf2cf5013e0e09aa7/R/objects.R
    """
    return df.loc[:, df.sum(axis = 0) > 200]

def filter_st(df, upper_thres_count = 1000):
    if df.shape[1] > upper_thres_count:
        cv = (df.std(axis = 0) / df.mean(axis = 0)).values
        df = df.loc[:, cv.sort_values(ascending = False).iloc[:upper_thres_count].index]
    return df

def normalize_st(df):
    df = df.astype(np.float32)
    cell_count = df.sum(axis = 1).values + 1e-5
    X = df.values
    N = df.median(axis = 1).values.reshape(-1,1)
    new_X = np.log(N * (X / cell_count.reshape(-1,1)) + 1).astype(np.float32)
    df[:] = new_X
    return df

def normalize_sc(df, scale_factor = 1e4):
    """
    ref 1 - https://github.com/satijalab/seurat/issues/3630
    ref 2 - https://github.com/satijalab/seurat/blob/763259d05991d40721dee99c9919ec6d4491d15e/R/preprocessing.R
    """
    df = df.astype(np.float32)
    cell_count = df.sum(axis = 1).values + 1e-5
    X = df.values
    new_X = np.log((X / cell_count.reshape(-1,1)) * scale_factor + 1).astype(np.float32)
    df[:] = new_X
    return df

def process_st(df):
    return filter_st(normalize_st(df))

def process_sc(df):
    return normalize_sc(df)

def run_spage(
    df_st: pd.DataFrame, 
    df_sc: pd.DataFrame, 
    train_list: list, 
    test_list: list
):
    from SpaGE.main import SpaGE
    df_sc = df_sc.loc[:, (df_sc.sum(axis=0) != 0)]
    df_sc = df_sc.loc[:, (df_sc.var(axis=0) != 0)]
    real = df_st.loc[:, test_list]
    predict = test_list
    feature = train_list
    pv = int(len(feature) / 2)
    if pv > 100:
        pv = 100
    df_st = df_st[feature]
    df_st2sc = SpaGE(df_st, df_sc, n_pv = pv, genes_to_predict = predict)
    result = df_st2sc[predict]
    return result, real

def run_gimvi(
    df_st: pd.DataFrame, 
    df_sc: pd.DataFrame, 
    train_list: list, 
    test_list: list
):
    """
    ref - https://docs.scvi-tools.org/en/0.8.0/user_guide/notebooks/gimvi_tutorial.html
    """
    import scvi
    import scanpy as sc
    from scvi.model import GIMVI
    import torch
    from torch.nn.functional import softmax, cosine_similarity, sigmoid
    df_sc = df_sc.loc[:, (df_sc.sum(axis = 0) != 0)]
    df_st_input = df_st.loc[:, [gene for gene in train_list if gene in df_sc.columns]] # spatial genes needs to be subset of seq genes

    ann_st = sc.AnnData(df_st_input)
    ann_sc = sc.AnnData(df_sc)
    sc.pp.filter_cells(ann_sc, min_counts = 1)
    sc.pp.filter_cells(ann_st, min_counts = 1)

    scvi.data.setup_anndata(ann_st)
    scvi.data.setup_anndata(ann_sc)
    model = GIMVI(ann_sc, ann_st)
    model.train(200)
  
    _, imputation = model.get_imputed_values(normalized=False)
    test_set = set(test_list)
    test_gene_index = [idx for idx, c in enumerate(df_sc.columns) if c in test_set]
    result = pd.DataFrame(data = imputation[:, test_gene_index], columns = df_sc.columns[test_gene_index]).loc[:,test_list]
    list_selected_index = ann_st.obs
    real = df_st.loc[[int(i) for i in ann_st.obs.index], test_list]
    del model
    return result, real

def run_ours(
    df_st: pd.DataFrame,
    df_sc: pd.DataFrame,
    train_list: list,
    test_list: list,
    save_name: str = None
):
    df_sc = df_sc.loc[:, (df_sc.sum(axis=0) != 0)]
    df_sc = df_sc.loc[:, (df_sc.var(axis=0) != 0)]
    df_sc = df_sc.astype(np.float32)

    df_st = df_st.astype(np.float32)
    
    print(f'SC: {df_sc.shape}, ST: {df_st.shape}')

    df_sc = np.log2(df_sc+1)
    df_st = np.log2(df_st+1)

    real = df_st.loc[:, test_list]

    in_features_source = len(train_list)
    in_features_target = df_sc.shape[1]
    
    in_features_source = len(train_list)
    in_features_target = df_sc.shape[1]
    # define configurations
    import yaml
    base_dir = '../'
    opts = yaml.safe_load(open(os.path.join(base_dir, 'options', 'base.yaml')))
    opts['train_opt']['device'] = 'cuda'
    opts['train_opt']['log_dir'] = os.path.join(base_dir, 'results/benchmark/ours/log') if save_name is None else os.path.join(base_dir, f'results/benchmark/{save_name}/ours/log')
    opts['train_opt']['epochs_enc_dec'] = 10
    opts['train_opt']['epochs'] = 10
    opts['model_opt']['enc_type_source'] = '1d_simple'
    opts['model_opt']['enc_type_target'] = '1d_simple'
    
    opts['exp_setting'] = {
        'fold_gene': 1,
        'fold_sample': 1,
        'cv_gene': 1,
        'cv_sample': 1,
        'cell_id': {
            'train': df_st.index.tolist(),
            'val': df_st.index.tolist(),
            'test': df_st.index.tolist()
        },
        'gene_names':{
            'val': [i for i in train_list if i in df_sc.columns],
            'test': [i for i in test_list if i in df_sc.columns],
            'source': df_st.columns.tolist(),
            'source_input': [c for c in df_st.columns.tolist() if c not in test_list],
            'target': df_sc.columns.tolist()
        }
    }
    
    train_opt = opts['train_opt']
    train_opt['batch_size'] = 512
    # define dataset
    dict_ds = {
        'target': {
            'train': ExpDataset(df_sc, df_sc.columns.tolist())
        }
    }
    dict_ds['source'] = {
        'train': ExpDataset(df_st, train_list, test_list),
        'val': ExpDataset(df_st, train_list, test_list),
        'test': ExpDataset(df_st, train_list, test_list)
    }
    # define dataloaders
    dict_dl = {}
    dict_dl['target'] = {
        'train': torch.utils.data.DataLoader(dict_ds['target']['train'], batch_size = train_opt['batch_size'], shuffle = True, drop_last = True, num_workers = train_opt['num_workers'])
    }
    dict_dl['source'] = {
        split: torch.utils.data.DataLoader(dict_ds['source'][split], batch_size = train_opt['batch_size'], shuffle = train_only, drop_last = train_only, num_workers = train_opt['num_workers'])
        for split, train_only in zip(['train', 'val', 'test'], [True, False, False])
    }
    # define trainer
    trainer = define_trainer(in_features_source, in_features_target, opts, df_st)
    trainer.train_enc_dec(dict_dl)
    result = trainer.train(dict_dl)
    del trainer
    return result, real
