# Goldfarbmuren, K.C., Jackson, N.D., Sajuthi, S.P., Dyjack, N., Li, K.S., Rios, C.L., Plender, E.G., Montgomery, M.T., Everman, J.L., Bratcher, P.E. and Vladar, E.K., 2020. Dissecting the cellular specificity of smoking effects and reconstructing lineages in the human airway epithelium. Nature communications, 11(1), p.2485.

require(Seurat)
require(Matrix)
working_dir <- file.path("R:/GeneAnalysis/data")

file_name <- "GSE134174_Processed_invitro"
save_name_prefix <- "gse134174_invitro"

### Load data
metadata <- read.delim(file.path(working_dir,
                                 paste(file_name, "_metadata.txt",
                                       sep = "")))
metadata$cell_type1 <- metadata$clust_ident
metadata$clust_ident <- NULL
donors <- metadata$donor
timepoints <- metadata$day
gset <- read.table(file.path(working_dir, paste(file_name, "_norm.txt",
                                                sep = "")),
                   stringsAsFactors = F)
features <- rownames(gset)
gset <- as.sparse(gset)
colnames(gset) <- rownames(metadata)
metadata <- metadata$cell_type1
metadata[metadata == "basal.colonies"] <- "Basal colonies"
metadata[metadata == "basal.confluent"] <- "Basal confluent"
metadata[metadata == "basal.subconfluent"] <- "Basal subconfluent proliferating"
metadata[metadata == "d.basal"] <- "Differentiating basal"
metadata[metadata == "p.basal"] <- "Basal proliferating within epithelium"
metadata[metadata == "secretory1"] <- "Secretory cells 1"
metadata[metadata == "secretory2"] <- "Secretory cells 2"
metadata[metadata == "ciliating.early"] <- "FOXN4+ early ciliating cells"
metadata[metadata == "ciliating.late"] <- "Later ciliating cells"
metadata[metadata == "ciliated"] <- "Mature ciliated cells"
metadata[metadata == "rare"] <- "Rare cells"

#########################################################
################### Basal vs non Basal ##################
#########################################################
save_name <- "_basal_vs_basallineages"
save_name <- paste(save_name_prefix, save_name, sep = "")
# group membership for all samples
# 0 (Basal): "Basal colonies", "Basal subconfluent proliferating", "Basal confluent",  "Differentiating basal", "Basal proliferating within epithelium"    
# 1 (non Basal): "Secretory cells 1", "Secretory cells 2", "Later ciliating cells", "Rare cells", "Mature ciliated cells", "FOXN4+ early ciliating cells" 
gsms <- c(0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1)
names(gsms) <- unique(metadata)
gsms <- gsms[metadata]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata
write.table(as.data.frame(celltypes),
            file = file.path(working_dir, paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir, paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir, paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(as.data.frame(donors),
            file = file.path(working_dir, paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(as.data.frame(timepoints),
            file = file.path(working_dir, paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset), file = file.path(working_dir, paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
################## Basal vs Secretory ###################
#########################################################
condition <- metadata %in% c("Basal colonies", "Basal subconfluent proliferating",
                             "Basal confluent", "Differentiating basal",
                             "Basal proliferating within epithelium",
                             "Secretory cells 1", "Secretory cells 2")
save_name <- "_basal_vs_secretory"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (Basal): "Basal colonies", "Basal subconfluent proliferating", "Basal confluent",  "Differentiating basal", "Basal proliferating within epithelium"
# 1 (Secretory): "Secretory cells 1", "Secretory cells 2"
gsms <- c(0, 0, 1, 0, 1, 0, 0)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata[condition]
write.table(as.data.frame(celltypes),
            file = file.path(working_dir, paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir, paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir, paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]),
            file = file.path(working_dir, paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]),
            file = file.path(working_dir, paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]),
        file = file.path(working_dir, paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
#################### Basal vs Ciliated ##################
#########################################################
condition <- metadata %in% c("Basal colonies", "Basal subconfluent proliferating",
                             "Basal confluent", "Differentiating basal",
                             "Basal proliferating within epithelium",
                             "Later ciliating cells", "Mature ciliated cells",
                             "FOXN4+ early ciliating cells")
save_name <- "_basal_vs_ciliated"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (Basal): "Basal colonies", "Basal subconfluent proliferating", "Basal confluent",  "Differentiating basal", "Basal proliferating within epithelium"
# 1 (Ciliated): "Later ciliating cells", "Mature ciliated cells", "FOXN4+ early ciliating cells"
gsms <- c(0, 0, 0, 0, 0, 1, 1, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata[condition]
write.table(as.data.frame(celltypes),
            file = file.path(working_dir, paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir, paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir, paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]),
            file = file.path(working_dir, paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]),
            file = file.path(working_dir, paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]),
        file = file.path(working_dir, paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
##################### Basal vs Rare #####################
#########################################################
condition <- metadata %in% c("Basal colonies", "Basal subconfluent proliferating",
                             "Basal confluent", "Differentiating basal",
                             "Basal proliferating within epithelium", "Rare cells")
save_name <- "_basal_vs_rare"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (Basal): "Basal colonies", "Basal subconfluent proliferating", "Basal confluent",  "Differentiating basal", "Basal proliferating within epithelium"
# 1 (Rare): "Rare cells"
gsms <- c(0, 0, 0, 0, 0, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata[condition]
write.table(as.data.frame(celltypes),
            file = file.path(working_dir, paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir, paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir, paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]),
            file = file.path(working_dir, paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]),
            file = file.path(working_dir, paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]),
        file = file.path(working_dir, paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
################# Secretory vs Ciliated ################# 
#########################################################
condition <- metadata %in% c("Secretory cells 1", "Secretory cells 2",
                             "Later ciliating cells", "Mature ciliated cells",
                             "FOXN4+ early ciliating cells")
save_name <- "_secretory_vs_ciliated"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (Secretory): "Secretory cells 1", "Secretory cells 2"
# 1 (Ciliated): "Later ciliating cells", "Mature ciliated cells", "FOXN4+ early ciliating cells"
gsms <- c(0, 0, 1, 1, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata[condition]
write.table(as.data.frame(celltypes),
            file = file.path(working_dir, paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir, paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir, paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]),
            file = file.path(working_dir, paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]),
            file = file.path(working_dir, paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]),
        file = file.path(working_dir, paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
################### Secretory vs Rare ################### 
#########################################################
condition <- metadata %in% c("Secretory cells 1", "Secretory cells 2", "Rare cells")
save_name <- "_secretory_vs_rare"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (Secretory): "Secretory cells 1", "Secretory cells 2"
# 1 (Rare): "Rare cells"
gsms <- c(0, 0, 1)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata[condition]
write.table(as.data.frame(celltypes),
            file = file.path(working_dir, paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir, paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir, paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]),
            file = file.path(working_dir, paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]),
            file = file.path(working_dir, paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]),
        file = file.path(working_dir, paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
#################### Ciliated vs Rare ################### 
#########################################################
condition <- metadata %in% c("Later ciliating cells", "Mature ciliated cells",
                             "FOXN4+ early ciliating cells", "Rare cells")
save_name <- "_ciliated_vs_rare"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (Ciliated): "Later ciliating cells", "Mature ciliated cells", "FOXN4+ early ciliating cells"
# 1 (Rare): "Rare cells"
gsms <- c(0, 1, 0, 0)
names(gsms) <- unique(metadata[condition])
gsms <- gsms[metadata[condition]]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save celltypes 
celltypes <- metadata[condition]
write.table(as.data.frame(celltypes),
            file = file.path(working_dir, paste(save_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save classes
classes <- as.numeric(sml)
write.table(as.data.frame(classes),
            file = file.path(working_dir, paste(save_name, "_classes.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save feature names
write.table(as.data.frame(features),
            file = file.path(working_dir, paste(save_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save donors
write.table(data.frame(donors = donors[condition]),
            file = file.path(working_dir, paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints[condition]),
            file = file.path(working_dir, paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
rownames(gset) <- features
writeMM(t(gset[, condition]),
        file = file.path(working_dir, paste(save_name, "_matrix.mtx", sep = "")))
