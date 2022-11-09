# TODO

# Differential expression analysis with limma
require(limma)
require(umap)

working_dir <- file.path("R:/GeneAnalysis/data")
file_name <- "pollen"
source("R:/GeneAnalysis/uhet/src/utility/create_sce.R")

### DATA
gset <- read.table(file.path(working_dir, "NBT_hiseq_linear_tpm_values.txt"))

### ANNOTATIONS
cell_type1 <- colnames(gset)
cell_type1[grepl("Hi_2338", cell_type1)] <- "2338"
cell_type1[grepl("Hi_2339", cell_type1)] <- "2339"
cell_type1[grepl("Hi_K562", cell_type1)] <- "K562"
cell_type1[grepl("Hi_BJ", cell_type1)] <- "BJ"
cell_type1[grepl("Hi_HL60", cell_type1)] <- "HL60"
cell_type1[grepl("Hi_iPS", cell_type1)] <- "hiPSC"
cell_type1[grepl("Hi_Kera", cell_type1)] <- "Kera"
cell_type1[grepl("Hi_GW21.2", cell_type1)] <- "GW21+3"
cell_type1[grepl("Hi_GW21", cell_type1)] <- "GW21"
cell_type1[grepl("Hi_NPC", cell_type1)] <- "NPC"
cell_type1[grepl("Hi_GW16", cell_type1)] <- "GW16"
cell_type2 <- colnames(gset)
cell_type2[grepl("Hi_K562", cell_type2) |
             grepl("Hi_HL60", cell_type2) |
             grepl("Hi_2339", cell_type2)] <- "blood"
cell_type2[grepl("Hi_BJ", cell_type2) |
             grepl("Hi_Kera", cell_type2) |
             grepl("Hi_2338", cell_type2)] <- "dermal"
cell_type2[grepl("Hi_iPS", cell_type2)] <- "pluripotent"
cell_type2[grepl("Hi_GW21.2", cell_type2) |
             grepl("Hi_GW21", cell_type2) |
             grepl("Hi_NPC", cell_type2) |
             grepl("Hi_GW16", cell_type2)] <- "neural"
classes <- data.frame(cell_type1 = cell_type1, cell_type2 = cell_type2)
rownames(classes) <- colnames(gset)

### SINGLECELLEXPERIMENT
sceset <- create_sce_from_normcounts(gset, classes)
