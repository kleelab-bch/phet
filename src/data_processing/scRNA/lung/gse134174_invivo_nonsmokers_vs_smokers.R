# Goldfarbmuren, K.C., Jackson, N.D., Sajuthi, S.P., Dyjack, N., Li, K.S., Rios, C.L., Plender, E.G., Montgomery, M.T., Everman, J.L., Bratcher, P.E. and Vladar, E.K., 2020. Dissecting the cellular specificity of smoking effects and reconstructing lineages in the human airway epithelium. Nature communications, 11(1), p.2485.

require(Seurat)
require(Matrix)
working_dir <- file.path("R:/GeneAnalysis/data")

file_name <- "GSE134174_Processed_invivo"
save_name_prefix <- "gse134174_invivo"

load_object <- function(file) {
  tmp <- new.env()
  load(file = file, envir = tmp)
  tmp[[ls(tmp)[1]]]
}

#########################################################
################ non Smokers vs Smokers ################# 
#########################################################
### Load data
gset <- load_object(file = file.path(working_dir, 
                                     paste(file_name, "_seurat.Rdata", 
                                           sep = "")))
metadata <- as.character(gset@meta.data$cluster_ident)
metadata[metadata == "Differentiating.basal"] <- "Differentiating basal"
metadata[metadata == "Proliferating.basal"] <- "Proliferating basal"
metadata[metadata == "Proteasomal.basal"] <- "Proteasomal basal"
metadata[metadata == "SMG.basal"] <- "SMG basal"
metadata[metadata == "KRT8.high"] <- "KRT8 intermediate"
metadata[metadata == "Mucus.secretory"] <- "Mucus secretory"
metadata[metadata == "SMG.secretory"] <- "SMG secretory"
metadata[metadata == "Ionocyte.tuft"] <- "Ionocyte+Tuft"
donors <- gset@meta.data$donor
timepoints <- as.character(gset@meta.data$Smoke_status)
subtypes <- as.character(gset@meta.data$subcluster_ident)
condition <- gset@meta.data$smoke
gset <- gset@assays$SCT@data
features <- rownames(gset)

save_name <- "_nonsmokers_vs_smokers"
save_name <- paste(save_name_prefix, save_name, sep = "")

# group membership for all samples
# 0 (Non smokers): "Non smokers"
# 1 (Smokers): "Smokers"
gsms <- c(1,1,0)
names(gsms) <- unique(condition)
gsms <- gsms[condition]
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
write.table(data.frame(donors = donors), 
            file = file.path(working_dir, 
                             paste(save_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save timepoints
write.table(data.frame(timepoints = timepoints), 
            file = file.path(working_dir, 
                             paste(save_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
writeMM(t(gset), 
        file = file.path(working_dir, 
                         paste(save_name, "_matrix.mtx", sep = "")))

#########################################################
#################### Rest of Analysis ################### 
#########################################################
### Load data
gset <- load_object(file = file.path(working_dir, 
                                     paste(file_name, "_seurat.Rdata", 
                                           sep = "")))
metadata <- as.character(gset@meta.data$cluster_ident)
metadata[metadata == "Differentiating.basal"] <- "Differentiating basal"
metadata[metadata == "Proliferating.basal"] <- "Proliferating basal"
metadata[metadata == "Proteasomal.basal"] <- "Proteasomal basal"
metadata[metadata == "SMG.basal"] <- "SMG basal"
metadata[metadata == "KRT8.high"] <- "KRT8 intermediate"
metadata[metadata == "Mucus.secretory"] <- "Mucus secretory"
metadata[metadata == "SMG.secretory"] <- "SMG secretory"
metadata[metadata == "Ionocyte.tuft"] <- "Ionocyte+Tuft"

# only include the following cell types
condition <- metadata %in% c("SMG basal", "Mucus secretory")
condition <- which(condition == "TRUE")
metadata <- metadata[condition]
donors <- gset@meta.data$donor[condition]
timepoints <- as.character(gset@meta.data$Smoke_status[condition])
subtypes <- as.character(gset@meta.data$subcluster_ident[condition])
condition_smokes <- gset@meta.data$smoke[condition]
gset <- gset@assays$SCT@data
features <- rownames(gset)
gset <- gset[, condition]

#########################################################
######### SMG basal (non) vs SMG basal (smokers) ######## 
#########################################################
condition <- metadata %in% c("SMG basal")
save_name <- "_smgbasalandnon_vs_smgbasalandsmokers"
save_name <- paste(save_name_prefix, save_name, sep = "")
gset_smgbasal <- gset[, condition]
condition_smokes_smgbasal <- condition_smokes[condition]

# group membership for all samples
# 0 (Non smokers): "SMG basal"
# 1 (Smokers): "SMG basal"
gsms <- c(1,1,0)
names(gsms) <- unique(condition_smokes_smgbasal)
gsms <- gsms[condition_smokes_smgbasal]
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
writeMM(t(gset_smgbasal), 
        file = file.path(working_dir, 
                         paste(save_name, "_matrix.mtx", sep = "")))
rm(gset_smgbasal)

#########################################################
## Mucus secretory (non) vs Mucus secretory (smokers) ## 
#########################################################
condition <- metadata %in% c("Mucus secretory")
save_name <- "_mucussecretoryandnon_vs_mucussecretoryandsmokers"
save_name <- paste(save_name_prefix, save_name, sep = "")
gset <- gset[, condition]
condition_smokes_mucus <- condition_smokes[condition]

# group membership for all samples
# 0 (Non smokers): "Mucus secretory"
# 1 (Smokers): "Mucus secretory"
gsms <- c(1,1,0)
names(gsms) <- unique(condition_smokes_mucus)
gsms <- gsms[condition_smokes_mucus]
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
writeMM(t(gset), 
        file = file.path(working_dir, 
                         paste(save_name, "_matrix.mtx", sep = "")))
