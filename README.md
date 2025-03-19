# MINERVA: **M**ultimodal **IN**tegration with self-sup**ERV**ised le**A**rning  
*A Generalizable Framework for Single-Cell Multiomics Analysis*

---

## 📖 Introduction  
**MINERVA** is a versatile framework for single-cell multimodal data integration, specifically optimized for CITE-seq data. It employs **six innovative self-supervised learning strategies** (grouped into bilevel masking, batch augmentation, cell fusion) to enable robust integrative analysis and cross-dataset generalization. 

### Key Capabilities  
✅ _De novo_ integration of heterogeneous multi-omics datasets into a unified latent space, especially for small-scale datasets  
✅ Dimensionality reduction  
✅ Impute missing features within and cross modalities  
✅ Batch effect removal  
✅ Zero-shot knowledge transfer to unseen datasets without additional training or fine-tuning  
✅ Instant cell-type identification  

---

## 🗂️ Training Data  
| Dataset<br>(Abbrev.)       | Species | Cells   | Proteins       | Batches | Accession ID |                Sample ratio: cell                 |
|       :--------------:     |  :---:  |  :---:  |    :-----:     |  :---:  |    :----:    |            :---------------------------           |
| Mouse CD45- dura mater<br>(DM) | Mouse   | 6,697   | 168            | 1       | GSE191075    | 10%: 664<br>20%: 1,336<br>50%: 3,346<br>100%: 6,697 |
| Spleen and lymph nodes<br>(SLN)    | Mouse   | 29,338  | SLN111:111<br>SLN208:208 | 4       | GSE150599    | 10%: 2,339<br>20%: 4,678<br>50%: 11,731<br>100%: 23,470 |
| Bone marrow mononuclear cell<br>(BMMC)| Human   | 90,261  | 134            | 12      | GSE194122    | 10%: 5,893<br>20%: 17,840<br>50%: 29,975<br>100%: 60,155 |
| Immune cells  across lineages and tissues<br>(IMC)| Human   | 190,877 | 268            | 15      | GSE229791    | -                                                 |

---

## ⚙️ Installation  
**System Requirements**  
- OS: Linux Ubuntu 18.04  
- Python 3.8.8 | R 4.1.0  

**Dependencies**  
```bash
# Create environment
conda create --name MINERVA python=3.8.8
conda activate MINERVA

# Install core packages
# All packages needed detailed in `others/Dependencies.txt`
pip install torch==2.0.0
conda install -c conda-forge r-seurat=4.3.0

# Clone repository
git clone https://github.com/labomics/MINERVA.git
cd MINERVA
```

---

## 🚀 Getting Started  
### 1. Data Preparation  
Conduct quality control on individual datasets and export the filtered data in **h5seurat** format for RNA and ADT modalities. Select variable features, generate corresponding expression matrices, and split them by cell for MINERVA input.<br>
<br>
Process demo data from `Example_data/`:  
```bash
# Quality control
Rscript Preparation/1_rna_adt_filter.R dm_sub10_demo.rds dm_sub10
Rscript Preparation/1_rna_adt_filter.R sln_sub10_demo.rds sln_sub10

# Feature selection
Rscript Preparation/2_combine_subsets.R dm_sub10_demo.rds dm_sub10
Rscript Preparation/2_combine_subsets.R sln_sub10_demo.rds sln_sub10

# Generate MINERVA inputs
python Preparation/3_split_exp.py --task dm_sub10
python Preparation/3_split_exp.py --task sln_sub10
```
For specific preprocessing needs, you may also choose Scanpy or Seurat. Once preprocessing is complete, split the matrices with **3_split_exp.py**.

### 2. MINERVA Application  
#### Scenario A: _De Novo_ Integration  
```bash
# Integration with SSL strategies
CUDA_VISIBLE_DEVICES=0 python MINERVA/run.py --task dm_sub10 --pretext mask
CUDA_VISIBLE_DEVICES=0 python MINERVA/run.py --task sln_sub10 --pretext mask noise downsample
# Cell fision strategies require ≥2 batches
CUDA_VISIBLE_DEVICES=0 python MINERVA/run.py --task bmmc_sub10 --pretext mask noise downsample fusion
```

**Output Extraction**  
This produces trained model states saved at each specified epoch. You can obtain the joint low-dimensional representation, the intra- and inter-modality imputed expression profiles, and the batch-corrected matrix with the following commands:
```bash
python MINERVA/run.py --task dm_sub10 --init_model sp_00000999 --actions predict_all
```
*Output paths*: `dm_sub10/e0/default/predict/sp_latest/subset_0/{z,x_impu,x_bc,x_trans}`<br>

#### Scenario B: Zero-Shot Generalization to Novel Queries  
We take two example:
1. trained MINERVA with two batches of SLN datasets, and test the transfer performance with the others batches.
```bash
# Split train/test datasets
mkdir -p ./result/preprocess/sln_sub10_train/{train,test}/

for dir in train test; do
    ln -sf ../../sln_sub10/feat ./result/preprocess/sln_sub10_train/$dir/
done

for i in 2 3; do
    ln -sf ../../sln_sub10/subset_$i ./result/preprocess/sln_sub10_train/train/subset_$((i-2))
done

ln -sf ../../sln_sub10/subset_{0,1} ./result/preprocess/sln_sub10_train/test/

# Train model
CUDA_VISIBLE_DEVICES=0 python MINERVA/run.py --task sln_sub10_train --pretext mask noise downsample --use_shm 2

# Knowledge transfer to novel batches
python MINERVA/run.py --task sln_sub10_transfer --ref sln_sub10_train --rf_experiment e0 \
--experiment e0 --init_model sp_latest --init_from_ref 1 --action predict_all  --use_shm 3
``` 
2. Reference atlas application to novel cross-tissues datasets
```bash
# Reference atlas setup
CUDA_VISIBLE_DEVICES=0 python MINERVA/run.py --task imc_ref --experiment all \
--pretext mask noise downsample fusion --use_shm 2

# Knowledge transfer to cross-tissues queries
python MINERVA/run.py --task imc_query --ref imc_ref --rf_experiment all \
--experiment all --init_model sp_latest --init_from_ref 1 --action predict_all --use_shm 3

```

### 3. Performance Evaluation  
The output from both scenarios includes input reconstructions, batch-corrected expression profiles, imputed matrices, cross-modality expression translations, and 34-dimensional joint embeddings (the first 32 dimensions for biological state and the last 2 dimensions for technical bias). These embeddings can be read using Python ("pd.read_csv") or R ("read.csv"), allowing for neighborhood graph computation and subsequent clustering with Anndata or Seurat. Quantitative evaluations of integration performance are also supported:<br>
```bash
# Batch correction & biological conservation
python Evaluation/benchmark_batch_bio.py

# Modality alignment
python Evaluation/benchmark_mod.py

# Comprehensive scoring
python Evaluation/combine_metrics.py
```

---

## 📌 Parameters  
| Argument           | Description                           | Options                          |
|--------------------|---------------------------------------|----------------------------------|
| `--pretext`        | SSL strategies                        | `mask`, `noise`, `downsample`, `fusion` |
| `--use_shm`        | dataset dir selected (all / train / test)| `1`-`3`                          |
| `--actions`        | Post-training operations              | `predict_all`, `predict_joint`, `et.al`   |

For full options:  
```bash
python MINERVA/run.py -h
```
