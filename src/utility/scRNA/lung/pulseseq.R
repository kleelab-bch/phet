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
metadata <- read.delim(file.path(working_dir, paste(file_name, "_md.txt", sep = "")))
metadata$cell_type1 <- metadata$res.2_named
metadata$res.2_named <- NULL
gset <- readRDS(file.path(working_dir, "GSE103354_PulseSeq_UMI_counts.rds"))
features <- gset@Dimnames[[1]]
gset <- CreateSeuratObject(counts = ps, min.cells = 0, min.genes = 0)
gset <- NormalizeData(object = gset, normalization.method = "LogNormalize",
                      scale.factor = 10000, display.progress = TRUE)
colnames(gset) <- rownames(metadata)
metadata <- metadata$cell_type1


#########################################################
##################### Use full data #####################
#########################################################
# The population of proliferating cells are predominantly basal cells
# group membership for all samples
# 0 (Basal): "Basal" and "Proliferating"
# 1 (non Basal): "Club", "Neuroendocrine", "Ciliated", "Goblet", "Ionocyte", and "Tuft"   
gsms <- c(0, 0, 1, 1, 1, 1, 1, 1)
names(gsms) <- unique(metadata)
gsms <- gsms[metadata]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
subtypes <- metadata
write.table(as.data.frame(subtypes), file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save metadata
classes <- as.numeric(sml)
write.table(as.data.frame(classes), file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(gset, file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))


#########################################################
################### Trim Basal cells ####################
#########################################################
# The population of proliferating cells are predominantly basal cells
condition <- !metadata %in% c("Basal", "Proliferating")
metadata <- metadata[condition]
file_name <- "pulseseq_club"

# group membership for all samples
# 0 (Club): "Club", "Ciliated", and "Goblet"
# 1 (non Club): Neuroendocrine", "Ionocyte", and "Tuft"
gsms <- c(0, 1, 0, 0, 1, 1)
names(gsms) <- unique(metadata)
gsms <- gsms[metadata]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
subtypes <- metadata
write.table(as.data.frame(subtypes), file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save metadata
classes <- as.numeric(sml)
write.table(as.data.frame(classes), file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
gset <- gset[, condition]
writeMM(gset, file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))


#########################################################
############### Trim Basal & Club cells ################# 
#########################################################
# The population of proliferating cells are predominantly basal cells
condition <- !metadata %in% c("Club", "Basal", "Proliferating")
metadata <- metadata[condition]
file_name <- "pulseseq_club_lineage"

# group membership for all samples
# 0 (Club lineage): "Goblet" and "Ciliated"
# 1 (non Club lineage): Neuroendocrine", "Ionocyte", and "Tuft"
gsms <- c(1, 0, 0, 1, 1)
names(gsms) <- unique(metadata)
gsms <- gsms[metadata]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes
subtypes <- metadata
write.table(as.data.frame(subtypes), file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save metadata
classes <- as.numeric(sml)
write.table(as.data.frame(classes), file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
gset <- gset[, condition]
writeMM(gset, file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))


#########################################################
########### Include Ciliated & Goblet cells #############
#########################################################
condition <- metadata %in% c("Ciliated", "Goblet")
temp_metadata <- metadata[condition]
file_name <- "pulseseq_goblet"

# group membership for all samples
# 0 (Ciliated): "Ciliated"
# 1 (Goblet): "Goblet"
gsms <- c(0, 1)
names(gsms) <- unique(temp_metadata)
gsms <- gsms[temp_metadata]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
subtypes <- temp_metadata
write.table(as.data.frame(subtypes), file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save temp_metadata
classes <- as.numeric(sml)
write.table(as.data.frame(classes), file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(gset[, condition], file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))

#########################################################
########## Include PNEC, Ionocyte, & Tuft cells #########
#########################################################
condition <- metadata %in% c("Neuroendocrine", "Ionocyte",
                             "Tuft")
metadata <- metadata[condition]
file_name <- "pulseseq_tuft"

# group membership for all samples
# 0 (PNEC and Ionocyte): "Neuroendocrine" and "Ionocyte"
# 1 (Tuft): "Tuft"
gsms <- c(0, 0, 1)
names(gsms) <- unique(metadata)
gsms <- gsms[metadata]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
subtypes <- metadata
write.table(as.data.frame(subtypes), file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save metadata
classes <- as.numeric(sml)
write.table(as.data.frame(classes), file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
gset <- gset[, condition]
writeMM(gset, file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))
