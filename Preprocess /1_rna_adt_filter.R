library(SeuratDisk)
source('utils.R')


base_path <- "./data/processed"
data_path <- "./Download_data/dura_mater"
dm_path <- pj(data_path, "dm_sub10.rds")
output_path <- pj(base_path, "dm_sub10/inputdata")
mkdir(output_path, remove_old = T)


# load data
dm = readRDS(dm_path)
dm@meta.data$orig.ident <- "WT" 
obj_split <- SplitObject(dm, split.by = "orig.ident")


for (batch in names(obj_split) ) {
    
    prt("Processing batch ", batch, " ...")
    output_dir <- pj(output_path, tolower(batch), "seurat")
    mkdir(output_dir, remove_old = T)
    
    rna_counts <- obj_split[[batch]]$RNA@counts
    adt_counts <- obj_split[[batch]]$ADT@counts
    # RNA
    rna <- gen_rna(rna_counts)
    # rna <- subset(rna, subset =
    #     nFeature_rna > 500 & nFeature_rna < 6000 & nCount_rna > 600 & nCount_rna < 40000 & percent.mt < 15)

    # ADT
    adt <- gen_adt(adt_counts)
    # adt <- subset(adt, subset = nCount_adt > 400 & nCount_adt < 20000)

    # Get intersected cells satisfying QC metrics of all modalities
    cell_ids <- Reduce(intersect, list(colnames(rna), colnames(adt)))
    rna <- subset(rna, cells = cell_ids)
    adt <- subset(adt, cells = cell_ids)
    
    # preprocess and save data
    preprocess(output_dir, rna = rna, adt = adt)
