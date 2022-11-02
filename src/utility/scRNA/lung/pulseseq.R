# Montoro, D.T., Haber, A.L., Biton, M., Vinarsky, V., Lin, B., Birket, S.E., Yuan, F., Chen, S., Leung, H.M., Villoria, J. and Rogel, N., 2018. A revised airway epithelial hierarchy includes CFTR-expressing ionocytes. Nature, 560(7718), pp.319-324.

# Differential expression analysis with limma
require(limma)
require(umap)
require(Seurat)
require(Matrix)
source("R:/GeneAnalysis/uhet/src/utility/create_sce.R")

working_dir <- file.path("R:/GeneAnalysis/data")
file_name <- "pulseseq"

### Load UMI count data from GEO
classes <- read.delim(file.path(working_dir, paste(file_name, "_md.txt", sep = "")))
classes$cell_type1 <- classes$res.2_named
classes$res.2_named <- NULL
gset <- readRDS(file.path(working_dir, "GSE103354_PulseSeq_UMI_counts.rds"))
features <- gset@Dimnames[[1]]
features[length(features) + 1] <- "class"
gset <- CreateSeuratObject(counts = ps, min.cells = 0, min.genes = 0)
gset <- NormalizeData(object = gset, normalization.method = "LogNormalize",
                      scale.factor = 10000, display.progress = TRUE)
colnames(gset) <- rownames(classes)
classes <- classes$cell_type1

# group membership for all samples
# 0 (basal): Basal
# 1 (non basal): "Proliferating", "Club", "Neuroendocrine", "Ciliated", "Goblet", "Ionocyte", and "Tuft"   
gsms <- c(0, 1, 1, 1, 1, 1, 1, 1)
names(gsms) <- unique(classes)
gsms <- gsms[classes]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# feature names
write.table(as.data.frame(features), file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# collect subtypes 
subtypes <- classes
write.table(as.data.frame(subtypes), file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save matrix data
gset <- rbind(gset, as.numeric(sml))
rownames(gset) <- features
writeMM(gset, file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))