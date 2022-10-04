# TODO

# Differential expression analysis with limma
require(limma)
require(umap)

working_dir <- file.path("R:/GeneAnalysis/data")
file_name <- "muraro"
source("R:/GeneAnalysis/uhet/src/utility/create_sce.R")

### DATA
gset <- read.table(file.path(working_dir, "GSE85241_cellsystems_dataset_4donors_updated.csv"), header = T, stringsAsFactors = F)
classes <- read.table(file.path(working_dir, "cell_type_annotation_Cels2016.csv"), stringsAsFactors = F)

### ANNOTATIONS
gset <- gset[,colnames(gset) %in% rownames(classes)]
classes <- classes[rownames(classes) %in% colnames(gset),,drop = FALSE]
gset <- gset[,order(colnames(gset))]
classes <- classes[order(rownames(classes)),,drop = FALSE]
colnames(classes) <- "cell_type1"
# format cell type names
classes$cell_type1[classes$cell_type1 == "duct"] <- "ductal"
classes$cell_type1[classes$cell_type1 == "pp"] <- "gamma"
tmp <- matrix(unlist(strsplit(rownames(classes),"[._]")), ncol=3, byrow=T)
classes$donor <- tmp[,1]
classes$batch <- tmp[,2]

### SINGLECELLEXPERIMENT
sceset <- create_sce_from_normcounts(gset, classes)
# use gene names as feature symbols
gene_names <- unlist(lapply(strsplit(rownames(sceset), "__"), "[[", 1))
rowData(sceset)$feature_symbol <- gene_names
# remove features with duplicated names
sceset <- sceset[!duplicated(rowData(sceset)$feature_symbol), ]