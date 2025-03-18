# MINERVA
Generalizable single-cell **M**ultimodal **IN**tegration with self sup**ERV**ised le**A**rning

# Introduction
MINERVA is a generalizable framework for single-cell data integration, designed for CITE-seq data analysis. It leverages six self-supervised learning strategies, grouped into bilevel masking, batch augmentation, and cell fusion, to enhance both integrative analysis and cross-dataset generalization of single-cell multimodal data. With MINERVA, users can:

1. De novo integrate heterogeneous multi-omics datasets into a shared latent space, especially for small-scale datasets.
2. Reduce dimensionality.
3. Impute missing features within and cross modalities.
4. Remove batch effects across different datasets.
5. Zero-shot knowledge transfer to unseen datasets without model retraining.
6. Instant identify uncharacterized cell types.

# Training Data
|             Dataset           |  Species  | Sample: Cell num  |  Protein num  |  Batch num  |  Accession ID  |     Sample ratio: cell num for _de novo_ training   |
|             :-----:           |   :---:   |      :----:       |    :----:     |    :----:   |    :----:      |                       :----                       |
|    Mouse CD45- dura mater<br>(DM)    |   mouse   |      6,697    |      168      |      1      |   GSE191075    |10%: 664<br>20%: 1,336<br>50%: 3,346<br>100%: 6,697|
|    Spleen and lymph nodes<br>(SLN)   |   mouse   |     29,338    |SLN111:111<br>SLN208:208|      4      |   GSE150599    |10%: 2,339<br>20%: 4,678<br>50%: 11,731<br>100%: 23,470|
|Bone marrow<br>mononuclear cell<br>(BMMC)|   human   |  90,261  |     134     |     12      |   GSE194122    |10%: 5,893<br>20%: 17,840<br>50%: 29,975<br>100%: 60,155|
|Immune cells across lineages and tissues<br>(IMC)|   human   |  190,877  |     268     |     15      |   GSE229791    |                           -                       |
 
# Installation
System: Linux Ubuntu 18.04
Programming language:<br>
- Python 3.8.8
- R 4.1.0

Main packages:
- Pytorch 2.0.0
- Seurat 4.3.0

```
# Create Environment
conda create --name MINERVA python=3.8.8
# Install from GitHub
git clone https://github.com/labomics/MINERVA.git
cd MINERVA
```

# Getting started
For facilitate quick startup, we provide a demo for preprocessing. You can choose test data in **Example_data** folder.

## 1.Prepare Data
Conduct quality control on individual datasets and export the filtered data in **h5seurat** format for RNA and ADT modalities. Select variable features, generate corresponding expression matrices, and split them by cell for MINERVA input.  
  
```
# Qulity control
Rscript Preparation/1_rna_adt_filter.R &
# Select ADT and Highly Variable Genes
Rscript Preparation/2_combine_subsets.R &
# Construct Input Files for MINERVA
python Preparation/3_split_exp.py &
```
For specific preprocessing needs, you may also choose Scanpy or Seurat. Once preprocessing is complete, split the matrices with **3_split_exp.py**.  

## 2.Apply MINERVA
We propose two application scenarios for MINERVA:
### (1) _De novo_ Integration  
   With prepared input data, execute the following script:
   ```
   CUDA_VISIBLE_DEVICES=0 python MINERVA/run.py --task dm_sub10 --pretext mask &
   # OR
   CUDA_VISIBLE_DEVICES=0 python MINERVA/run.py --task sln_sub10 --pretext mask noise downsample &
   # OR
   CUDA_VISIBLE_DEVICES=0 python MINERVA/run.py --task bmmc_sub10 --pretext mask noise downsample fusion &
   ```
   - pretext：SSL strategies used.<br>
 **Note: The dataset needs more than two batches to use the fusion strategy.**
   <br>
   This produces trained model states saved at each specified epoch. You can obtain the joint low-dimensional representation, the intra- and inter-modality imputed expression profiles, and the batch-corrected matrix with the following commands:
   ```
   python MINERVA/run.py --task dm_sub10 --init_model sp_00000999 --actions predict_all &
   ```
   The multimodal integration outputs are systematically stored in a hierarchical directory structure following the pattern, such as: _dm_sub10/e0/default/predict/sp_latest/subset_0/{z,x_impu,x_bc,x_trans}_.<br>
   <br>
   The integrated performance was subsequently assessed using the **3.Evaluate Performance** framework.
   <br>
### (2) Generalization to Novel Queries  
   Pre-trained model from **scenarios (1)** can serve as a reference atlas seamlessly integrating unseen query data and accurately transfers cell-type labels without the need for de novo integration or fine-tuning.<br>
   
   ```
   # Set up Large-scale Reference Atlas
   CUDA_VISIBLE_DEVICES=0 python MINERVA/run.py --task imc_ref --experiment all --pretext mask noise downsample fusion --use_shm 2 &

   # Perform Knowledge Transfer
   python MINERVA/run.py --task imc_query --ref imc_ref --rf_experiment all --experiment all --init_model sp_latest --init_from_ref 1 --action predict_all --use_shm 3 &
   ```
   The output will be generated as in scenario 1 and can also be evaluated for generalizated performance.  
   
## 3.Evaluate Performance
The output from both scenarios includes input reconstructions, batch-corrected expression profiles, imputed matrices, cross-modality expression translations, and 34-dimensional joint embeddings (the first 32 dimensions for biological state and the last 2 dimensions for technical bias). These embeddings can be read using Python ("pd.read_csv") or R ("read.csv"), allowing for neighborhood graph computation and subsequent clustering with Anndata or Seurat. Quantitative evaluations of integration performance are also supported:
```
# Benchmark of Biological Conservation and Batch Correction
python Evalution/benchmark_batch_bio.py

# Benchmark of Modality Alignment
python Evalution/benchmark_mod.py

# Comprehensive Scoring
python Evalution/combine_metrics.py
```
 
Here, we only provide a set of commonly used MINERVA parameters for reference. For further parameter options, use ```python MINERVA.py -h```.
