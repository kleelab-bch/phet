# Plasschaert, L.W., Zilionis, R., Choo-Wing, R., Savova, V., Knehr, J., Roma, G., Klein, A.M. and Jaffe, A.B., 2018. A single-cell atlas of the airway epithelium reveals the CFTR-rich pulmonary ionocyte. Nature, 560(7718), pp.377-381.

# Differential expression analysis with limma
require(limma)
require(umap)
require(Seurat)
require(Matrix)
source("R:/GeneAnalysis/uhet/src/utility/create_sce.R")

working_dir <- file.path("R:/GeneAnalysis/data")
file_name <- "plasschaert_mouse_secretory_vs_rare"

### Load data
metadata <- read.table(file.path(working_dir,
                                 paste("GSE102580_meta_mouse.tsv", sep = "")),
                       header = TRUE, sep = "\t", row.names = 1,
                       check.names = FALSE, stringsAsFactors = FALSE)
donors <- metadata$mouse_id
timepoints <- metadata$timepoint
metadata$cell_type1 <- metadata$clusters_Fig2
metadata <- metadata$cell_type1
# remove Basal cells 
condition <- metadata %in% c("Secretory", "Pre-ciliated", "Ciliated", "Brush",
                             "PNEC", "Ionocytes")

metadata <- metadata[condition]
# markers
df <- read.table(file.path(working_dir,
                           paste("plasschaert_mouse_all_markers.csv", sep = "")),
                 header = TRUE, sep = ",", row.names = 1, check.names = FALSE,
                 stringsAsFactors = FALSE)
enriched_features <- df$EnrichedIn %in% c("Secretory", "Pre-ciliated", "Ciliated", "Brush",
                                          "PNEC", "Ionocytes")
features <- rownames(df)[enriched_features]
write.table(as.data.frame(features),
            file = file.path(working_dir, paste(file_name, "_markers.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)
donors <- donors[condition]
write.table(as.data.frame(donors),
            file = file.path(working_dir, paste(file_name, "_donors.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)
timepoints <- timepoints[condition]
write.table(as.data.frame(timepoints),
            file = file.path(working_dir, paste(file_name, "_timepoints.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)
# already normalized
gset <- read.table(file.path(working_dir,
                             paste("GSE102580_normalized_counts_mouse.tsv", sep = "")),
                   header = TRUE, sep = "\t", row.names = 1,
                   check.names = FALSE, stringsAsFactors = FALSE)
features <- rownames(gset)
gset <- gset[, condition]
gset <- as.data.frame(t(gset))

# group membership for all samples
# 0 (Secretory cells): "Secretory"
# 1 (Ciliated and Rare cells): "Pre-ciliated", "Ciliated", "Brush", "PNEC", and "Ionocytes" 
gsms <- c(0, 1, 1, 1, 1, 1)
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
            file = file.path(working_dir,
                             paste(file_name, "_feature_names.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save data
df <- data.matrix(gset)
df <- as(df, "dgCMatrix")
writeMM(df, file = file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))
remove(df)