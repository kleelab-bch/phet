# Montoro, D.T., Haber, A.L., Biton, M., Vinarsky, V., Lin, B., Birket, S.E., Yuan, F., Chen, S., Leung, H.M., Villoria, J. and Rogel, N., 2018. A revised airway epithelial hierarchy includes CFTR-expressing ionocytes. Nature, 560(7718), pp.319-324.

# Differential expression analysis with limma
require(limma)
require(umap)
require(Seurat)
require(Matrix)
source("R:/GeneAnalysis/phet/src/data_processing/create_sce.R")

working_dir <- file.path("R:/GeneAnalysis/data")
file_name <- "pulseseq"

### Load UMI count data from GEO
metadata <- read.delim(file.path(working_dir, paste(file_name, "_md.txt", sep = "")))
metadata$cell_type1 <- metadata$res.2_named
metadata$res.2_named <- NULL
donors <- metadata$mouse
timepoints <- metadata$timepoint
gset <- readRDS(file.path(working_dir, "GSE103354_PulseSeq_UMI_counts.rds"))
features <- gset@Dimnames[[1]]
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
write.table(as.data.frame(subtypes), 
            file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes), 
            file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), 
            file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(as.data.frame(donors), 
            file = file.path(working_dir, paste(file_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(as.data.frame(timepoints), 
            file = file.path(working_dir, paste(file_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset), 
        file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))

#########################################################
############### Basal vs Club & its Lineage #############
#########################################################
condition <- metadata %in% c("Basal", "Proliferating", 
                             "Club", "Ciliated", "Goblet")
file_name <- "pulseseq_basal_vs_clubandlineage"

# group membership for all samples
# 0 (Basal): "Basal" and "Proliferating"
# 1 (Club and Club Lineage): "Club", "Ciliated", and "Goblet"
gsms <- c(0, 0, 1, 1, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
subtypes <- metadata[condition]
write.table(as.data.frame(subtypes), 
            file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes), 
            file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), 
            file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]), 
            file = file.path(working_dir, paste(file_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]), 
            file = file.path(working_dir, paste(file_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]), 
        file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))

#########################################################
################ Basal vs Non club lineage ##############
#########################################################
condition <- metadata %in% c("Basal", "Proliferating", 
                             "Neuroendocrine", "Ionocyte", 
                             "Tuft")
file_name <- "pulseseq_basal_vs_nonclublineage"

# group membership for all samples
# 0 (Basal): "Basal" and "Proliferating"
# 1 (Non Club Lineage): "PNEC", "Ionocyte", and "Tuft"
gsms <- c(0, 0, 1, 1, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
subtypes <- metadata[condition]
write.table(as.data.frame(subtypes), 
            file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes), 
            file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), 
            file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]), 
            file = file.path(working_dir, paste(file_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]), 
            file = file.path(working_dir, paste(file_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]), 
        file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))

#########################################################
################ Basal vs Neuroendocrine ################
#########################################################
condition <- metadata %in% c("Basal", "Proliferating", 
                             "Neuroendocrine")
file_name <- "pulseseq_basal_vs_neuroendocrine"

# group membership for all samples
# 0 (Basal): "Basal" and "Proliferating"
# 1 (PNEC): "Neuroendocrine"
gsms <- c(0, 0, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
subtypes <- metadata[condition]
write.table(as.data.frame(subtypes), 
            file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes), 
            file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), 
            file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]), 
            file = file.path(working_dir, paste(file_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]), 
            file = file.path(working_dir, paste(file_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]), 
        file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))

#########################################################
##################### Basal vs Tuft #####################
#########################################################
condition <- metadata %in% c("Basal", "Proliferating", 
                             "Tuft")
file_name <- "pulseseq_basal_vs_tuft"

# group membership for all samples
# 0 (Basal): "Basal" and "Proliferating"
# 1 (Tuft): "Tuft"
gsms <- c(0, 0, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
subtypes <- metadata[condition]
write.table(as.data.frame(subtypes), 
            file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes), 
            file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), 
            file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]), 
            file = file.path(working_dir, paste(file_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]), 
            file = file.path(working_dir, paste(file_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]), 
        file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))

#########################################################
################### Basal vs Ionocyte ###################
#########################################################
condition <- metadata %in% c("Basal", "Proliferating", 
                             "Ionocyte")
file_name <- "pulseseq_basal_vs_ionocyte"

# group membership for all samples
# 0 (Basal): "Basal" and "Proliferating"
# 1 (Ionocyte): "Ionocyte"
gsms <- c(0, 0, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
subtypes <- metadata[condition]
write.table(as.data.frame(subtypes), 
            file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes), 
            file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), 
            file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]), 
            file = file.path(working_dir, paste(file_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]), 
            file = file.path(working_dir, paste(file_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]), 
        file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))

#########################################################
################ Club & Lineage vs Rare #################
#########################################################
# The population of proliferating cells are predominantly basal cells
condition <- metadata %in% c("Club", "Ciliated", "Goblet", 
                             "Neuroendocrine", "Ionocyte", 
                             "Tuft")
file_name <- "pulseseq_clubandlineage_vs_rare"

# group membership for all samples
# 0 (Club/Lineage): "Club", "Ciliated", and "Goblet"
# 1 (Rare): "PNEC", "Ionocyte", and "Tuft"
gsms <- c(0, 1, 0, 0, 1, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
subtypes <- metadata[condition]
write.table(as.data.frame(subtypes), 
            file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes), 
            file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), 
            file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]), 
            file = file.path(working_dir, paste(file_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]), 
            file = file.path(working_dir, paste(file_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]), 
        file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))

#########################################################
################## Club vs Club Lineage #################
#########################################################
# The population of proliferating cells are predominantly basal cells
condition <- metadata %in% c("Club", "Ciliated", "Goblet")
file_name <- "pulseseq_club_vs_clublineage"

# group membership for all samples
# 0 (Club): "Club"
# 1 (Club Lineage): "Ciliated", and "Goblet"
gsms <- c(0, 1, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
subtypes <- metadata[condition]
write.table(as.data.frame(subtypes), 
            file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes), 
            file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), 
            file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]), 
            file = file.path(working_dir, paste(file_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]), 
            file = file.path(working_dir, paste(file_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]), 
        file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))

#########################################################
################### Club vs Ciliated ####################
#########################################################
# The population of proliferating cells are predominantly basal cells
condition <- metadata %in% c("Club", "Ciliated")
file_name <- "pulseseq_club_vs_ciliated"

# group membership for all samples
# 0 (Club): "Club"
# 1 (Ciliated): "Ciliated"
gsms <- c(0, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
subtypes <- metadata[condition]
write.table(as.data.frame(subtypes), 
            file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes), 
            file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), 
            file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]), 
            file = file.path(working_dir, paste(file_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]), 
            file = file.path(working_dir, paste(file_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]), 
        file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))

#########################################################
#################### Club vs Goblet #####################
#########################################################
# The population of proliferating cells are predominantly basal cells
condition <- metadata %in% c("Club", "Goblet")
file_name <- "pulseseq_club_vs_goblet"

# group membership for all samples
# 0 (Club): "Club"
# 1 (Goblet): "Goblet"
gsms <- c(0, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
subtypes <- metadata[condition]
write.table(as.data.frame(subtypes), 
            file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes), 
            file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), 
            file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]), 
            file = file.path(working_dir, paste(file_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]), 
            file = file.path(working_dir, paste(file_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]), 
        file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))

#########################################################
################## Ciliated vs Goblet ###################
#########################################################
condition <- metadata %in% c("Ciliated", "Goblet")
file_name <- "pulseseq_ciliated_vs_goblet"

# group membership for all samples
# 0 (Ciliated): "Ciliated"
# 1 (Goblet): "Goblet"
gsms <- c(0, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
subtypes <- metadata[condition]
write.table(as.data.frame(subtypes), 
            file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes), 
            file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), 
            file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]), 
            file = file.path(working_dir, paste(file_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]), 
            file = file.path(working_dir, paste(file_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]), 
        file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))

#########################################################
############## Tuft vs PNEC & Ionocytes #################
#########################################################
condition <- metadata %in% c("Neuroendocrine", "Ionocyte",
                             "Tuft")
file_name <- "pulseseq_tuft_vs_pnecandionocyte"

# group membership for all samples
# 0 (Tuft): "Tuft"
# 1 (PNEC and Ionocytes): "Neuroendocrine" and "Tuft"
gsms <- c(1, 1, 0)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
subtypes <- metadata[condition]
write.table(as.data.frame(subtypes), 
            file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes), 
            file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), 
            file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]), 
            file = file.path(working_dir, paste(file_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]), 
            file = file.path(working_dir, paste(file_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]), 
        file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))

#########################################################
################## Tuft vs Ionocytes ####################
#########################################################
condition <- metadata %in% c("Ionocyte", "Tuft")
file_name <- "pulseseq_tuft_vs_ionocyte"
gset < gset[, condition]
# group membership for all samples
# 0 (Tuft): "Tuft"
# 1 (Ionocytes): "Ionocytes"
gsms <- c(1, 0)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
subtypes <- metadata[condition]
write.table(as.data.frame(subtypes), 
            file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes), 
            file = file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features), 
            file = file.path(working_dir, paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]), 
            file = file.path(working_dir, paste(file_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]), 
            file = file.path(working_dir, paste(file_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]), 
        file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))
