import torch
import torchvision.transforms.functional as TF
import os, pickle, numpy as np, pandas as pd, scanpy as sc

def load_dataframe(source_name, data_dir, file_format = 'pickle'):
    def load(file_path):
        if file_format == 'pickle':
            return pd.read_pickle(file_path)
        else:
            return pd.read_csv(file_path)
    suffix = 'pkl' if file_format == 'pickle' else 'csv'
    if source_name == 'gse':
        file_path = os.path.join(data_dir, source_name, 'GSE131907_reference_sc_data-001.' + suffix)
    elif source_name == 'nanostring':
        file_path = os.path.join(data_dir, source_name, 'Lung9_Rep1_exprMat_file.' + suffix)
    elif source_name == 'osmfish':
        file_path = os.path.join(data_dir, source_name, 'osmFISH_data_ST.' + suffix)
    elif source_name == 'merfish':
        file_path = os.path.join(data_dir, source_name, 'MERFISH_data_ST.' + suffix)
    elif source_name == 'seqfish':
        file_path = os.path.join(data_dir, source_name, 'seqFISH_data_ST.' + suffix)
    elif source_name == 'starmap':
        file_path = os.path.join(data_dir, source_name, 'STARmap_data_ST.' + suffix)
    elif source_name == 'allen_ssp':
        file_path = os.path.join(data_dir, source_name, 'AllenSSp_data_SC.' + suffix)
    elif source_name == 'allen_visp':
        file_path = os.path.join(data_dir, source_name, 'AllenVISp_data_SC.' + suffix)
    elif source_name == 'moffit':
        file_path = os.path.join(data_dir, source_name, 'Moffit_data_SC.' + suffix)
    elif source_name == 'zeisel':
        file_path = os.path.join(data_dir, source_name, 'Zeisel_data_SC.' + suffix)
    return load(file_path).astype(np.float32)

def process_dataframe(df, min_count_gene, min_count_cell, min_density_gene, min_density_cell, gene_selection_count, clip_outlier, normalization, genes_to_keep = None):
    """
    Arguments:
        genes_to_keep (list): keep the necessary genes to keep (no filter out for these genes)
    Order or processing
        1. Drop genes/cells by minimum density required
        2. gene selection by count
        3. clip outliers
        4. normalization
    """
    # 1. drop gene/cell
    df = df.astype(np.float32)
    
    # minimum count
    if min_count_gene is not None and min_count_gene > 0: 
        gene_count = df.sum(axis = 0)
        valid_gene_slice = gene_count > min_count_gene
        # Keep the important genes
        if genes_to_keep is not None:
            valid_gene_slice = set(valid_gene_slice[valid_gene_slice].index.tolist()).union(set(genes_to_keep))
            valid_gene_slice = [c for c in df.columns if c in valid_gene_slice]
        df = df.loc[:, valid_gene_slice]
    if min_count_cell is not None and min_count_cell > 0:
        cell_count = df.sum(axis = 1)
        valid_cell_slice = cell_count > min_count_cell
        df = df.loc[valid_cell_slice, :]
    # minimum density
    if min_density_gene is not None and min_density_gene > 0:
        gene_density = (df > 0).sum(axis = 0) / df.shape[0]
        valid_gene_slice = gene_density > min_density_gene
        # Keep the important genes
        if genes_to_keep is not None:
            valid_gene_slice = set(valid_gene_slice[valid_gene_slice].index.tolist()).union(set(genes_to_keep))
            valid_gene_slice = [c for c in df.columns if c in valid_gene_slice]
        df = df.loc[:,valid_gene_slice]
    if min_density_cell is not None and min_density_cell > 0:
        cell_density = (df > 0).sum(axis = 1) / df.shape[1]
        valid_cell_slice = cell_density > min_density_cell
        df = df.loc[valid_cell_slice, :]

    # 2. gene selection by count
    if gene_selection_count is not None and gene_selection_count > 0:

        adata = sc.AnnData(df)
        # Change to flavor = cell_ranger for merfish
        sc.pp.highly_variable_genes(adata, n_top_genes = gene_selection_count, subset = True, flavor = 'seurat_v3')
        adata_hvg = adata[:, adata.var['highly_variable']]
        valid_genes = adata_hvg.var.index
        
        if genes_to_keep is not None:
            genes_to_keep = (set(genes_to_keep).intersection(set(valid_genes)))
            valid_genes = set(valid_genes).union(set(genes_to_keep))
            valid_genes = [c for c in df.columns if c in valid_genes]
        valid_genes = list(valid_genes)
        df = df.loc[:,valid_genes]
        
    # clip outliers
    if clip_outlier:
        for col in df.columns:
            mean, std = df.loc[df[col] > 0, col].agg(['mean', 'std'])
            upper_limit = mean + 2 * std
            df[col] = df[col].astype(np.float64)
            df.loc[df[col] > upper_limit, col] = upper_limit
            df[col] = df[col].astype(np.float32)
    # normalization
    if normalization == 'ours':
        adata = sc.AnnData(df)
        sc.pp.normalize_total(adata, target_sum = 1e4)
        sc.pp.log1p(adata)
        df = pd.DataFrame(index = adata.obs.index, columns = adata.var.index)
        df.iloc[:,:] = adata.X
        df = df.astype(np.float32)
    elif normalization == 'sqrt':
        df = np.sqrt(df)
    elif normalization == 'log2':
        df = np.log2(df + 1)
    elif normalization == 'standard':
        df = (df - df.mean(axis = 0)) / (df.std(axis = 0) + 1e-5)
    elif normalization == 'spage':
        df = np.log(((df/df.sum(axis=0))*1000000) + 1)

    return df.astype(np.float32)

class ExpDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        gene_exp_input: list,
        gene_exp_eval: list = None,
    ):
        self.df = df.copy()
        self.gene_names = df.columns.tolist()
        self.gene_exp_input = gene_exp_input
        self.gene_exp_eval = gene_exp_eval
        self.list_index = df.index.tolist()
    def __len__(self):
        return len(self.list_index)
    def __getitem__(self, idx):
        index = self.list_index[idx]
        input = self.df.loc[index, self.gene_exp_input].values
        if (self.gene_exp_eval is not None) and (len(self.gene_exp_eval) != 0):
            eval = self.df.loc[index, self.gene_exp_eval].values
            return_data = {
                'input': input,
                'eval': eval,
                'index': index
            }
        else: 
            return_data = {
            'input': input,
            'index': index
        }

        return return_data

class CompositeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        dict_image_data: dict, # has WSI, spatial information
        gene_exp_input: list,
        gene_exp_eval: list = None
    ):
        self.df = df
        self.dict_image_data = dict_image_data
        self.gene_names = df.columns.tolist()
        self.gene_exp_input = gene_exp_input
        self.gene_exp_eval = gene_exp_eval
        self.list_index = df.index.tolist()
    def __len__(self):
        return len(self.list_index)
    def __getitem__(self, idx):
        index = self.list_index[idx]
        meta = self.dict_image_data['meta'][index]
        fov = int(index.split('-')[0].split('_')[-1])
        cell_id = int(index.split('-')[1])
        cx,cy,height,width = meta['cx'], meta['cy'], meta['height'], meta['width']
        center_h = self.dict_image_data['label'][fov].shape[0] - cy
        center_w = cx
        h_lower = center_h - height // 2
        h_upper = center_h + (height - height // 2)
        w_lower = center_w - width // 2
        w_upper = center_w + (width - width // 2)
        
        image = self.dict_image_data['image'][fov][:,h_lower:h_upper, w_lower:w_upper].copy()
        label = self.dict_image_data['label'][fov][h_lower:h_upper, w_lower:w_upper].copy()
        input = self.df.loc[index, self.gene_exp_input].values
        if self.gene_exp_eval:
            eval = self.df.loc[index, self.gene_exp_eval].values
            return {
                'input': input,
                'eval': eval,
                'index': index,
                'image': image,
                'label': label
            }
        return {
                'input': input,
                'index': index,
                'image': image,
                'label': label
        }
    
def normalize_minmax_numpy(image, min_val = 0, max_val = 1, fill_val = 0):
    masked = np.ma.masked_equal(image, 0.0, copy = True)
    mins = masked.min(axis = (-1,-2)).data[None,:,None,None]
    maxs = masked.max(axis = (-1,-2)).data[None,:,None,None]
    masked = (((masked - mins) / (maxs - mins)) * (max_val - min_val)) + min_val
    image = masked.data
    image[masked.mask] = fill_val
    return image

def composite_collate_fn(data, target_size = (224,224), exclude_background = True, normalize_method = 'minmax1'):
    output = {}
    if exclude_background:
        image = [t['image'].astype(np.float32) * (t['label'] == int(t['index'].split('-')[1])) for t in data]
    else:
        image = [t['image'].astype(np.float32) for t in data]
    if normalize_method == 'minmax1':
        image = torch.cat([TF.center_crop(torch.Tensor(normalize_minmax_numpy(img)), target_size) for img in image], axis = 0)
    else:
        image = torch.cat([TF.center_crop(torch.Tensor(img), target_size).unsqueeze(0) for img in image], axis = 0)
    
    output['image'] = image
    for key in ['input', 'eval']:
        output[key] = torch.stack([torch.Tensor(t[key]) for t in data])
    output['index'] = [t['index'] for t in data]
    return output
