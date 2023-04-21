# Goldfarbmuren, K.C., Jackson, N.D., Sajuthi, S.P., Dyjack, N., Li, K.S., Rios, C.L., Plender, E.G., Montgomery, M.T., Everman, J.L., Bratcher, P.E. and Vladar, E.K., 2020. Dissecting the cellular specificity of smoking effects and reconstructing lineages in the human airway epithelium. Nature communications, 11(1), p.2485.

require(Seurat)
require(Matrix)
working_dir <- file.path("R:/GeneAnalysis/data")

file_name <- "GSE134174_Processed_invivo"
save_name_prefix <- "gse134174_invivo_nonsmokers"

load_object <- function(file) {
  tmp <- new.env()
  load(file = file, envir = tmp)
  tmp[[ls(tmp)[1]]]
}

### Load data
gset <- load_object(file = file.path(working_dir,
                                     paste(file_name, "_seurat.Rdata",
                                           sep = "")))
condition <- gset@meta.data$smoke %in% c("never")
condition <- which(condition == "TRUE")
donors <- gset@meta.data$donor[condition]
timepoints <- as.character(gset@meta.data$Smoke_status[condition])
subtypes <- as.character(gset@meta.data$subcluster_ident[condition])
metadata <- as.character(gset@meta.data$cluster_ident[condition])
metadata[metadata == "Differentiating.basal"] <- "Differentiating basal"
metadata[metadata == "Proliferating.basal"] <- "Proliferating basal"
metadata[metadata == "Proteasomal.basal"] <- "Proteasomal basal"
metadata[metadata == "SMG.basal"] <- "SMG basal"
metadata[metadata == "KRT8.high"] <- "KRT8 intermediate"
metadata[metadata == "Mucus.secretory"] <- "Mucus secretory"
metadata[metadata == "SMG.secretory"] <- "SMG secretory"
metadata[metadata == "Ionocyte.tuft"] <- "Ionocyte+Tuft"
gset <- gset@assays$SCT@data
gset <- gset[, condition]
features <- rownames(gset)

#########################################################
################ Basal vs Basal lineages ################ 
#########################################################
save_name <- "_basal_vs_basallineages"
save_name <- paste(save_name_prefix, save_name, sep = "")
# The population of proliferating cells are predominantly basal cells
# group membership for all samples
# 0 (Basal): "Differentiating basal", "Proliferating basal", "SMG basal", "Proteasomal basal"
# 1 (Basal lineages): "Mucus secretory", "KRT8 intermediate", "Ciliated", "Ionocyte+Tuft", "SMG secretory", "PNEC"
gsms <- c(1, 1, 1, 0, 1, 0, 0, 0, 1, 1)
names(gsms) <- unique(metadata)
gsms <- gsms[metadata]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata
write.table(as.data.frame(celltypes),
            file = file.path(working_dir,
                             paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir,
                             paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir,
                             paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(as.data.frame(donors),
            file = file.path(working_dir,
                             paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(as.data.frame(timepoints),
            file = file.path(working_dir,
                             paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
writeMM(t(gset), file = file.path(working_dir,
                                  paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
################## Basal vs Secretory ###################
#########################################################
condition <- metadata %in% c("Differentiating basal", "SMG basal",
                             "Proliferating basal", "Proteasomal basal",
                             "KRT8 intermediate", "Mucus secretory",
                             "SMG secretory")
save_name <- "_basal_vs_secretory"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (Basal): "Differentiating basal", "SMG basal", "Proliferating basal", "Proteasomal basal"
# 1 (Secretory/KRT8): "KRT8 intermediate", "Mucus secretory", "SMG secretory"
gsms <- c(1, 1, 0, 0, 0, 0, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata[condition]
write.table(as.data.frame(celltypes),
            file = file.path(working_dir,
                             paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir,
                             paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir,
                             paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
writeMM(t(gset[, condition]),
        file = file.path(working_dir,
                         paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
############## SMG Basal vs SMG Secretory ###############
#########################################################
condition <- metadata %in% c("SMG basal", "SMG secretory")
save_name <- "_smgbasal_vs_smgsecretory"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (SMG basal): "SMG basal"
# 1 (SMG secretory): "SMG secretory"
gsms <- c(0, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata[condition]
write.table(as.data.frame(celltypes),
            file = file.path(working_dir,
                             paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir,
                             paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir,
                             paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
writeMM(t(gset[, condition]),
        file = file.path(working_dir,
                         paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
#################### Basal vs Ciliated ##################
#########################################################
condition <- metadata %in% c("Differentiating basal", "SMG basal",
                             "Proliferating basal", "Proteasomal basal",
                             "Ciliated")
save_name <- "_basal_vs_ciliated"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (Basal): "Differentiating basal", "SMG basal", "Proliferating basal", "Proteasomal basal"
# 1 (Ciliated): "Ciliated"
gsms <- c(1, 0, 0, 0, 0)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata[condition]
write.table(as.data.frame(celltypes),
            file = file.path(working_dir,
                             paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir,
                             paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir,
                             paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
writeMM(t(gset[, condition]),
        file = file.path(working_dir,
                         paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
##################### Basal vs Rare #####################
#########################################################
condition <- metadata %in% c("Differentiating basal", "SMG basal",
                             "Proliferating basal", "Proteasomal basal",
                             "Ionocyte+Tuft", "PNEC")
save_name <- "_basal_vs_rare"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (Basal): "Differentiating basal", "SMG basal", "Proliferating basal", "Proteasomal basal"
# 1 (Rare): "Ionocyte+Tuft", "PNEC"
gsms <- c(0, 1, 0, 0, 0, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata[condition]
write.table(as.data.frame(celltypes),
            file = file.path(working_dir,
                             paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir,
                             paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir,
                             paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
writeMM(t(gset[, condition]),
        file = file.path(working_dir,
                         paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
################# Secretory vs Ciliated ################# 
#########################################################
condition <- metadata %in% c("Mucus secretory", "SMG secretory",
                             "Ciliated")
save_name <- "_secretory_vs_ciliated"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (Secretory): "Mucus secretory", "SMG secretory"
# 1 (Ciliated): "Ciliated"
gsms <- c(0, 1, 0)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata[condition]
write.table(as.data.frame(celltypes),
            file = file.path(working_dir,
                             paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir,
                             paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir,
                             paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
writeMM(t(gset[, condition]),
        file = file.path(working_dir,
                         paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
################### Secretory vs Rare ################### 
#########################################################
condition <- metadata %in% c("Mucus secretory", "SMG secretory",
                             "Ionocyte+Tuft", "PNEC")
save_name <- "_secretory_vs_rare"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (Secretory): "Mucus secretory", "SMG secretory"
# 1 (Rare): "Ionocyte+Tuft", "PNEC"
gsms <- c(0, 1, 0, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata[condition]
write.table(as.data.frame(celltypes),
            file = file.path(working_dir,
                             paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir,
                             paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir,
                             paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
writeMM(t(gset[, condition]),
        file = file.path(working_dir,
                         paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
############ Mucus secretory vs SMG secretory ########### 
#########################################################
condition <- metadata %in% c("Mucus secretory", "SMG secretory")
save_name <- "_mucussecretory_vs_smgsecretory"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (Mucus Secretory): "Mucus secretory"
# 1 (SMG Secretory): "SMG secretory"
gsms <- c(0, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata[condition]
write.table(as.data.frame(celltypes),
            file = file.path(working_dir,
                             paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir,
                             paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir,
                             paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
writeMM(t(gset[, condition]),
        file = file.path(working_dir,
                         paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
########## Mucus secretory vs KRT8 intermediate ######### 
#########################################################
condition <- metadata %in% c("Mucus secretory", "KRT8 intermediate")
save_name <- "_mucussecretory_vs_krt8"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (Mucus Secretory): "Mucus secretory"
# 1 (KRT8 intermediate): "KRT8 intermediate"
gsms <- c(0, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata[condition]
write.table(as.data.frame(celltypes),
            file = file.path(working_dir,
                             paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir,
                             paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir,
                             paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
writeMM(t(gset[, condition]),
        file = file.path(working_dir,
                         paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
#################### Ciliated vs Rare ################### 
#########################################################
condition <- metadata %in% c("Ciliated", "Ionocyte+Tuft", "PNEC")
save_name <- "_ciliated_vs_rare"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (Ciliated): "Ciliated"
# 1 (Rare): "Ionocyte+Tuft", "PNEC"
gsms <- c(0, 1, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata[condition]
write.table(as.data.frame(celltypes),
            file = file.path(working_dir,
                             paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir,
                             paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir,
                             paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
writeMM(t(gset[, condition]),
        file = file.path(working_dir,
                         paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
################# Ionocyte+Tuft vs PNEC ################# 
#########################################################
condition <- metadata %in% c("Ionocyte+Tuft", "PNEC")
save_name <- "_ionocyteandtuft_vs_pnec"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (Ionocyte+Tuft): "Ionocyte+Tuft"
# 1 (PNEC): "PNEC"
gsms <- c(0, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata[condition]
write.table(as.data.frame(celltypes),
            file = file.path(working_dir,
                             paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir,
                             paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir,
                             paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]),
            file = file.path(working_dir,
                             paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
writeMM(t(gset[, condition]),
        file = file.path(working_dir,
                         paste(save_name, "_matrix.mtx", sep = "")))