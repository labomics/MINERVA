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
