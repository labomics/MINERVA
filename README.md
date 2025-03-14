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
|               Dataset                |  Species  | Sample: Cell num  |  Protein num  |  Batch num  |  Accession ID  |        Sample ratio: cell num for training        |
|                :-----:               |   :---:   |      :----        |    :----:     |    :----:   |    :----:      |                       :----                       |
|    Mouse CD45- dura mater<br>(DM)    |   mouse   |        6,697      |      168      |      1      |   GSE191075    |10%: 664<br>20%: 1,336<br>50%: 3,346<br>100%: 6,697|
|    Spleen and lymph nodes<br>(SLN)   |   mouse   |SLN111-D1: 8,435<br>SLN111-D2: 6,574<br>SLN208-D1: 7,969; SLN208-D2: 6,360<br>Total: 29,338|SLN111:111<br>SLN208:208|      4      |   GSE150599    |10%: 2,339<br>20%: 4,678<br>50%: 11,731<br>100%: 23,470|
|Bone marrow mononuclear cell<br>(BMMC)|   human   |s1d1: 5,227; s1d2: 4,978; s1d3: 6,106;<br>s2d1: 10,465; s2d4: 5,584; s2d5: 9,122;<br>s3d1: 9,521; s3d6: 11,035; s3d7: 11,473;<br>s4d1: 5,456; s4d8: 3,929; s4d9: 7,365;<br>Total: 90,261|     134     |     12      |   GSE194122    |10%: 5,893<br>20%: 17,840<br>50%: 29,975<br>100%: 60,155|
|Immune cells across lineages and tissues<br>(IMC)|   human   |D496_BLD: 7,154; D496_BOM: 11,553;<br>D496_JEL: 6,446; D496_JLP: 8,187;<br>D496_LLN: 16,116; D496_LNG: 15,626;<br>D496_SPL: 16,422;<br>D503_BAL: 18,601; D503_BLD: 16,367;<br>D503_BOM: 18,148; D503_JEL: 8,712;<br>D503_JLP: 11,098; D503_LLN: 13,228;<br>D503_LNG: 10,531; D503_SPL: 12,688;<br>Total: 190,877|     268     |     15      |   GSE229791    |                           -                       |
 
# Running MINERVA
For facilitate quick startup, we provide a demo for preprocessing. You can use a test data in Example_data folder.

## 1.Prepare Data
Conduct quality control on individual datasets and export the filtered data in **h5seurat** format for RNA and ADT modalities. Select variable features, generate corresponding expression matrices, and split them by cell for MINERVA input.  
  
```
# Qulity control
Rscript Preprocess/1_rna_adt_filter.R &
# Select ADT and Highly Variable Genes
Rscript Preprocess/2_combine_subsets.R &
# Construct Input Files for MINERVA
python Preprocess/3_split_exp.py &
```
For specific preprocessing needs, you may also choose Scanpy or Seurat. Once preprocessing is complete, split the matrices with 3_split_exp.py.  

## 2.Apply MINERVA
We propose two application scenarios for MINERVA:
### (1) De novo Integration  
   With prepared input data, execute the following script:
   ```
   CUDA_VISIBLE_DEVICES=0 python run.py --task dm --experiment mask --pretext mask &
   # OR
   CUDA_VISIBLE_DEVICES=0 python run.py --task sln --experiment mask --pretext mask noise downsample &
   # OR
   CUDA_VISIBLE_DEVICES=0 python run.py --task bmmc --experiment mask --pretext mask noise downsample fusion &
   ```
   This produces trained model states saved at each specified epoch. You can obtain the joint low-dimensional representation, the intra- and inter-modality imputed expression profiles, and the batch-corrected matrix with the following commands:
   ```
   python run.py --task dm --experiment mask --init_model sp_latest --actions predict_all &
   ```
### (2) Generalization to Novel Queries  
   Pre-trained model from **scenario 1** can serve as a reference atlas seamlessly integrating unseen query data and accurately transfers cell-type labels without the need for de novo integration or fine-tuning:
   ```
   # Set up Large-scale Reference Atlas
   CUDA_VISIBLE_DEVICES=0 python run.py --task imc_ref --experiment all --pretext mask noise downsample fusion --use_shm 2 &

   # Perform Knowledge Transfer
   python run.py --task imc_query --ref imc_ref --rf_experiment all --experiment all --init_model sp_latest --init_from_ref 1 --action predict_all --use_shm 3 &
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
