"""
Most of codes are based on benchmarking paper
"""

import sys, os, time, pickle, importlib.util, numpy as np, pandas as pd
from .train_utils import define_trainer
from .data_utils import ExpDataset
import torch


def _load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _add_local_path(*parts):
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), *parts)
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
    return path


def _resolve_third_party_path(*parts):
    repo_root = os.path.dirname(os.path.dirname(__file__))
    candidates = [
        os.path.join(repo_root, *parts),
        os.path.join(os.path.dirname(repo_root), "fastSpaGene", *parts),
    ]
    for path in candidates:
        if os.path.exists(path):
            if path not in sys.path:
                sys.path.insert(0, path)
            return path
    return candidates[0]


def _resolve_use_gpu(device):
    if not torch.cuda.is_available():
        return False
    if device is None:
        return True
    if str(device).lower() == "cpu":
        return False
    return device


def _resolve_device(device_str):
    if isinstance(device_str, str) and device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def _fallback_spatial_coords(index, width=256):
    n = len(list(index))
    rows = np.arange(n, dtype=np.float32) // width
    cols = np.arange(n, dtype=np.float32) % width
    return np.stack([cols, rows], axis=1)


def _sanitize_spatial_coords(index, coords):
    fallback = _fallback_spatial_coords(index)
    if coords is None:
        return fallback

    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape != fallback.shape:
        raise ValueError(
            f"coords shape {coords.shape} does not match expected shape {fallback.shape}"
        )

    valid_mask = np.isfinite(coords).all(axis=1)
    if bool(valid_mask.all()):
        return coords

    coords = coords.copy()
    valid_coords = coords[valid_mask]
    if valid_coords.size == 0:
        return fallback

    x_offset = float(np.nanmax(valid_coords[:, 0])) + 1024.0
    y_offset = float(np.nanmax(valid_coords[:, 1])) + 1024.0
    coords[~valid_mask] = fallback[~valid_mask] + np.array([x_offset, y_offset], dtype=np.float32)
    return coords


def _make_anndata(df, coords=None):
    import scanpy as sc

    adata = sc.AnnData(
        df.to_numpy(dtype=np.float32, copy=False),
        obs=pd.DataFrame(index=pd.Index(df.index.astype(str))),
        var=pd.DataFrame(index=pd.Index(df.columns.astype(str))),
    )
    adata.obs["batch"] = "batch0"
    adata.obs["names"] = adata.obs_names.astype(str)
    if coords is not None:
        adata.obsm["spatial"] = np.asarray(coords, dtype=np.float32)
    return adata


def pick_stdiff_sizes(n_genes, train_opt):
    batch_size = train_opt.get("stdiff_batch_size")
    hidden_size = train_opt.get("stdiff_hidden_size")
    if n_genes < 512:
        default_batch_size, default_hidden_size = 2048, 512
    elif n_genes < 1024:
        default_batch_size, default_hidden_size = 512, 1024
    else:
        default_batch_size, default_hidden_size = 512, 2048

    if batch_size is None:
        batch_size = default_batch_size
    if hidden_size is None:
        hidden_size = default_hidden_size

    return int(batch_size), int(hidden_size)

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


def run_vista(
    df_st: pd.DataFrame,
    df_sc: pd.DataFrame,
    train_list: list,
    test_list: list,
    train_opt: dict,
    coords=None,
    return_real: bool = True,
):
    _resolve_third_party_path("third_party", "VISTA")
    try:
        from vista import GIMVI_GCN
    except Exception as exc:
        raise ImportError(
            "VISTA baseline import failed. Install or vendor the `vista` package under "
            "`SpaGene/third_party/VISTA`."
        ) from exc

    train_list = [g for g in train_list if g in df_st.columns and g in df_sc.columns]
    test_list = [g for g in test_list if g in df_sc.columns]
    if len(train_list) == 0:
        raise ValueError("VISTA baseline received no shared training genes.")
    if len(test_list) == 0:
        raise ValueError("VISTA baseline received no target genes to predict.")

    coords = _sanitize_spatial_coords(df_st.index, coords)
    adata_st = _make_anndata(df_st.loc[:, train_list].astype(np.float32), coords=coords)
    adata_sc = _make_anndata(df_sc.astype(np.float32))

    GIMVI_GCN.setup_anndata(adata_st, batch_key="batch", obs_names="names")
    GIMVI_GCN.setup_anndata(adata_sc, batch_key="batch")

    vista_batch_size = int(train_opt.get("vista_batch_size", train_opt.get("batch_size", 128)))
    model = GIMVI_GCN(
        adata_sc,
        adata_st,
        n_latent=int(train_opt.get("vista_n_latent", 32)),
        neighbor_size=int(train_opt.get("vista_neighbor_size", 20)),
        correlation_const=bool(train_opt.get("vista_correlation_const", False)),
    )
    model.train(
        max_epochs=int(train_opt.get("vista_epochs", 200)),
        batch_size=vista_batch_size,
        use_gpu=_resolve_use_gpu(train_opt.get("device")),
    )

    imputed = model.get_imputed_values(normalized=False, batch_size=vista_batch_size)[0]
    sc_var_names = adata_sc.var_names.astype(str).tolist()
    gene_to_idx = {g: idx for idx, g in enumerate(sc_var_names)}
    eval_test = [g for g in test_list if g in gene_to_idx]
    if len(eval_test) == 0:
        raise ValueError("VISTA baseline produced no evaluable target genes.")

    pred = pd.DataFrame(
        np.asarray(imputed[:, [gene_to_idx[g] for g in eval_test]], dtype=np.float32),
        index=adata_st.obs_names.astype(str),
        columns=eval_test,
    )
    if not return_real:
        return pred

    real = df_st.copy()
    real.index = real.index.astype(str)
    real.columns = real.columns.astype(str)
    return pred, real.loc[pred.index, eval_test]


def run_sprefine(
    df_st: pd.DataFrame,
    observed_genes: list,
    hidden_genes: list,
    train_opt: dict,
    return_real: bool = True,
):
    _resolve_third_party_path("third_party", "sprefine")
    try:
        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping
        from sprefine.model import Cell_Encoder, Gene_Encoder, sprefine as SpRefineModel
    except Exception as exc:
        raise ImportError(
            "spRefine baseline import failed. Install or vendor `sprefine` plus its dependencies "
            "under `SpaGene/third_party/sprefine`."
        ) from exc

    emb_path = train_opt.get("sprefine_gene_emb_path")
    if not emb_path:
        raise ValueError("spRefine requires `train_opt.sprefine_gene_emb_path`.")

    observed_genes = [g for g in observed_genes if g in df_st.columns]
    hidden_genes = [g for g in hidden_genes if g in df_st.columns]
    all_genes = list(dict.fromkeys(observed_genes + hidden_genes))

    if len(observed_genes) == 0:
        raise ValueError("spRefine baseline received no observed genes.")
    if len(hidden_genes) == 0 and return_real:
        raise ValueError("spRefine baseline received no hidden genes to predict.")

    import anndata as ad

    if str(emb_path).endswith(".h5ad"):
        adata_emb = ad.read_h5ad(emb_path)
        if "add_id" not in adata_emb.obs.columns:
            raise ValueError("spRefine embedding .h5ad is missing `obs['add_id']`.")
        emb = adata_emb.X.toarray() if hasattr(adata_emb.X, "toarray") else np.asarray(adata_emb.X)
        emb_df = pd.DataFrame(emb, index=adata_emb.obs["add_id"].astype(str))
    elif str(emb_path).endswith((".pkl", ".pickle")):
        emb_df = pickle.load(open(emb_path, "rb"))
    else:
        emb_df = pd.read_csv(emb_path, index_col=0)

    emb_df.index = emb_df.index.astype(str)
    observed_genes = [g for g in observed_genes if g in emb_df.index]
    hidden_genes = [g for g in hidden_genes if g in emb_df.index]
    all_genes = list(dict.fromkeys(observed_genes + hidden_genes))
    if len(observed_genes) == 0:
        raise ValueError("spRefine embedding leaves no observed genes for this fold.")

    gene_emb = torch.tensor(emb_df.loc[all_genes].to_numpy(dtype=np.float32), dtype=torch.float32)
    train_index = torch.tensor([all_genes.index(g) for g in observed_genes], dtype=torch.long)

    x_obs = np.ascontiguousarray(df_st.loc[:, observed_genes].to_numpy(dtype=np.float32, copy=False))
    device = _resolve_device(train_opt.get("device", "cpu"))
    batch_size = int(train_opt.get("sprefine_batch_size", train_opt.get("batch_size", 2048)))
    epochs = int(train_opt.get("sprefine_epochs", 1000))
    lr = float(train_opt.get("sprefine_lr", 1e-4))
    hidden_dim = int(train_opt.get("sprefine_hidden_dim", 64))
    nonneg = bool(train_opt.get("sprefine_nonneg", False))
    patience = int(train_opt.get("sprefine_patience", 100))
    val_fraction = float(train_opt.get("sprefine_val_fraction", 0.25))
    seed = int(train_opt.get("seed", 2024))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df_st))
    if len(perm) < 4:
        train_idx = perm
        val_idx = perm[:0]
    else:
        val_n = max(1, int(round(len(perm) * val_fraction)))
        val_n = min(val_n, len(perm) - 1)
        val_idx = perm[:val_n]
        train_idx = perm[val_n:]

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(np.ascontiguousarray(x_obs[train_idx])),
        torch.from_numpy(np.ascontiguousarray(x_obs[train_idx])),
    )
    val_ds = None
    if len(val_idx) > 0:
        val_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(np.ascontiguousarray(x_obs[val_idx])),
            torch.from_numpy(np.ascontiguousarray(x_obs[val_idx])),
        )

    num_workers = int(train_opt.get("num_workers", 4))
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            drop_last=False,
        )

    model = SpRefineModel(
        encoder1=Cell_Encoder(input_dim=len(observed_genes), hidden_dim=hidden_dim),
        encoder2=Gene_Encoder(input_dim=gene_emb.shape[1], hidden_dim=hidden_dim),
        gene_emb=gene_emb.to(device),
        train_index=train_index.to(device),
        eta=lr,
        nonneg=nonneg,
    )
    trainer_kwargs = dict(
        max_epochs=epochs,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    if val_loader is not None:
        trainer_kwargs["callbacks"] = [EarlyStopping(monitor="val_loss", mode="min", patience=patience)]

    Trainer(**trainer_kwargs).fit(model, train_loader, val_loader)

    model = model.to(device)
    model.gene_emb = model.gene_emb.to(device)
    model.train_index = model.train_index.to(device)
    model.eval()
    with torch.no_grad():
        pred_all = model(torch.as_tensor(x_obs, dtype=torch.float32, device=device)).cpu().numpy()

    output_genes = hidden_genes if len(hidden_genes) > 0 else all_genes
    output_idx = [all_genes.index(g) for g in output_genes]
    pred_df = pd.DataFrame(pred_all[:, output_idx], index=df_st.index.astype(str), columns=output_genes)
    if not return_real:
        return pred_df

    real_df = df_st.copy()
    real_df.index = real_df.index.astype(str)
    return pred_df, real_df.loc[:, output_genes]


def run_stdiff(
    df_st: pd.DataFrame,
    df_sc: pd.DataFrame,
    train_opt: dict,
):
    st_diff_root = _resolve_third_party_path("third_party", "stDiff")
    runner_path = os.path.join(st_diff_root, "fastspagene_stdiff_runner.py")
    if not os.path.exists(runner_path):
        raise ImportError(
            "stDiff baseline runner not found. Add it under `SpaGene/third_party/stDiff/fastspagene_stdiff_runner.py`."
        )

    runner = _load_module_from_path("spagene_stdiff_runner", runner_path)
    import scanpy as sc

    batch_size, hidden_size = pick_stdiff_sizes(df_st.shape[1], train_opt)
    adata_spatial = sc.AnnData(
        df_st.to_numpy(dtype=np.float32, copy=False),
        obs=pd.DataFrame(index=pd.Index(df_st.index.astype(str))),
        var=pd.DataFrame(index=pd.Index(df_st.columns.astype(str))),
    )
    adata_seq = sc.AnnData(
        df_sc.to_numpy(dtype=np.float32, copy=False),
        obs=pd.DataFrame(index=pd.Index(df_sc.index.astype(str))),
        var=pd.DataFrame(index=pd.Index(df_sc.columns.astype(str))),
    )
    pred = runner.run_stdiff(
        adata_spatial,
        adata_seq,
        device=str(train_opt.get("device", "cuda:0")),
        batch_size=batch_size,
        hidden_size=hidden_size,
        step=int(train_opt.get("stdiff_step", 1500)),
        epoch=int(train_opt.get("stdiff_epoch", 900)),
        noise_std=float(train_opt.get("stdiff_noise_std", 10.0)),
        head=int(train_opt.get("stdiff_head", 16)),
        rand=int(train_opt.get("seed", 0)),
        n_splits=int(train_opt.get("cv_gene", 5)),
    )
    pred.index = pred.index.astype(str)
    pred.columns = pred.columns.astype(str)
    return pred

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
