# SpaGene
SpaGene: A Deep Adversarial Framework for Spatial Gene Imputation

## Overview
We introduce a novel deep learning framework, SpaGene, which integrates scRNA-seq data and spatial transcriptomics data through an encoder-decoder architecture with translators, and discriminators, aiming to impute missing gene expression profiles in spatial transcriptomics. The model compresses spatial transcriptomics data into a latent space, translates it into scRNA-seq domain, and then decodes it to ultimately imputed unmeasured genes.

SpaGene is built using pytorch
Test on: Ubuntu 22.04.5 LTS, NVIDIA A40 GPU, AMD EPYC 74F3 24-Core Processor, 64 GB, python 3.10.14, CUDA environment(cuda 12.8)

## Table of Contents

- [Requirements](#requirements)
- [Folder structure](#folder-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Benchmarking](#benchmarking)

## Requirements
Required modules can be installed via requirements.txt under the project root
```
pip install -r requirements.txt
```
Check the list in following section:

```
anndata==0.10.8
arrow==1.3.0
h5py==3.11.0
jupyter==1.1.1
jupyterlab==4.2.5
lightning==2.1.4
lightning-bolts @ git+https://github.com/PytorchLightning/lightning-bolts.git@2c4602aa684e7b90e7ffdcea1d3f93a20f9c2ead
lightning-lite==1.8.0
lightning-utilities==0.11.6
llvmlite==0.43.0
loompy==3.0.7
matplotlib==3.9.1
matplotlib-inline==0.1.7
notebook==7.2.2
numba==0.60.0
numpy==1.26.3
opencv-python==4.10.0.84
pandas==2.2.3
pytorch-lightning==2.4.0
rpds-py==0.20.0
scanpy==1.10.2
scikit-image==0.24.0
scikit-learn==1.5.1
scikit-misc==0.5.1
scipy==1.14.0
scvi-tools==1.1.5
seaborn==0.13.2
statsmodels==0.14.2
stdlib-list==0.10.0
tensorboard==2.17.0
tensorboard-data-server==0.7.2
tifffile==2024.7.24
torch==2.4.0+cu124
torchmetrics==0.11.4
torchvision==0.19.0+cu124
tqdm==4.66.5
```
## Folder structure
```
├── codes
├── data
│   └── paired_datasets
│        └── allen_ssp
│              ├── AllenSSp_data_SC.csv
│              ├── AllenSSp_data_SC.csv          
│        └── allen_visp
│              ├── AllenVISp_data_SC.csv
│              ├── AllenVISp_data_SC.csv
│        └── gse
│              ├── GSE131907_reference_sc_data-001.csv
│              ├── GSE131907_reference_sc_data-001.pkl
│        └── nanostring
│              ├── Lung9_Rep1_exprMat_file.csv
│              └── Lung9_Rep1_exprMat_file.pkl
│              └── matched_annotation_all.csv
│        └── merfish
│              ├── MERFISH_data_ST.csv
│              └── MERFISH_data_ST.pkl
│        └── moffit
│              ├── Moffit_data_SC.csv
│              └── Moffit_data_SC.pkl
│        └── osmfish
│              ├── osmFISH_data_ST.csv
│              ├── osmFISH_data_ST.pkl
│        └── seqfish
│              ├── seqFISH_data_ST.pkl
│              ├── seqFISH_data_ST.pkl
│        └── starmap
│              ├── STARmap_data_ST.csv
│              ├── STARmap_data_ST.pkl
│        └── zeisel
│              ├── Zeisel_data_SC.csv
│              ├── Zeisel_data_SC.pkl
├── results
├── results_gimvi
├── results_spage
├── requirements.txt
├── generate_benchmark.py
├── train.py
```
## Installation

Download SpaGene:
```
git clone https://github.com/asbudhkar/SpaGene
```
## Dataset

### NanoString CosMx SMI Lung 9 rep1
The dataset can be download [here](https://nanostring.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/)
### GSE
The dataset can be download [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907)
### MERFISH Moffit, osmFISH AllenSSp, osmFISH AllenVISp, osmFISH Zeisel, seqFISH AllenVISp, STARmap AllenVISp 
The datasets can be download from public repository [here](https://zenodo.org/records/3967291)

## Training

#### 1. MERFISH Moffit
```
python3 train.py --exp merfish2moffit
```

#### 2. Nanostring GSE
```
python3 train.py --exp nano2gse
```
#### 3. osmFISH AllenSSp
```
python3 train.py --exp osmfish2allenssp
```

#### 4. osmFISH AllenVISp
```
python3 train.py --exp osmfish2allenvisp
```
#### 5. osmFISH Zeisel 
```
python3 train.py --exp osmfish2zeisel
```

#### 6. seqFISH AllenVISp
```
python3 train.py --exp seqfish2allenvisp
```
#### 7. STARmap AllenVISp
```
python3 train.py --exp starmap2allenvisp
```

## Benchmarking

#### 1. GimVI
```
python3 generate_benchmark.py --exp benchmark_base_gimvi
```
#### 2. SpaGE
```
python3 generate_benchmark.py --exp benchmark_base_spage
```
## Cite

Please cite our paper if you use this code in your own work

