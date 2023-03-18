# Plasschaert, L.W., Zilionis, R., Choo-Wing, R., Savova, V., Knehr, J., Roma, G., Klein, A.M. and Jaffe, A.B., 2018. A single-cell atlas of the airway epithelium reveals the CFTR-rich pulmonary ionocyte. Nature, 560(7718), pp.377-381.

# Differential expression analysis with limma
require(limma)
require(umap)
require(Seurat)
require(Matrix)
working_dir <- file.path("R:/GeneAnalysis/data")

file_name <- "plasschaert_mouse_basalandplus_vs_secretoryandkrt4"

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
condition <- metadata %in% c("Basal", "Cycling Basal (homeostasis)", 
                              "Cycling Basal (regeneration)", "Secretory",
                              "Krt4/13+")
metadata <- metadata[condition]
# markers
df <- read.table(file.path(working_dir,
                           paste("plasschaert_mouse_all_features.csv", sep = "")),
                 header = TRUE, sep = ",", row.names = 1, check.names = FALSE,
                 stringsAsFactors = FALSE)
enriched_features <- df$EnrichedIn %in% c("Basal", "Cycling Basal (homeostasis)", 
                                           "Cycling Basal (regeneration)", "Secretory",
                                           "Krt4/13+")
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
# 0 (Basal cells): "Basal", "Cycling Basal (homeostasis)", and "Cycling Basal (regeneration)"
# 1 (Secretory and Krt4/13+ cells): "Secretory" and "Krt4/13+"                
gsms <- c(1, 0, 1, 0, 0)
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

# assign samples to groups and set up design matrix
gs <- factor(sml)
groups <- make.names(c("Control", "Case"))
levels(gs) <- groups
gset$group <- gs
design <- model.matrix(~group + 0, gset)
colnames(design) <- levels(gs)

gset <- gset[, !(names(gset) %in% "group")]
gset <- t(gset)
fit <- lmFit(gset, design)  # fit linear model

# set up contrasts of interest and recalculate model coefficients
cts <- paste(groups[1], groups[2], sep = "-")
cont.matrix <- makeContrasts(contrasts = cts, levels = design)
fit2 <- contrasts.fit(fit, cont.matrix)

# compute statistics and table of top significant genes
fit2 <- eBayes(fit2, 0.01)
tT <- topTable(fit2, adjust = "fdr", sort.by = "B", number = 10000)
temp <- rownames(tT)
rownames(tT) <- NULL
tT <- cbind("ID" = temp, tT)
write.table(tT, file = file.path(working_dir, paste(file_name, "_diff_features.csv",
                                                    sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# Visualize and quality control test results.
# Build histogram of P-values for all genes. Normal test
# assumption is that most genes are not differentially expressed.
tT2 <- topTable(fit2, adjust = "fdr", sort.by = "B", number = Inf)
hist(tT2$adj.P.Val, breaks = 100, col = "grey", border = "white", xlab = "P-adj",
     ylab = "Number of genes", main = "P-adj value distribution")

# summarize test results as "up", "down" or "not expressed"
dT <- decideTests(fit2, adjust.method = "fdr", p.value = 0.01)
# Venn diagram of results
vennDiagram(dT, circle.col = palette())

# create Q-Q plot for t-statistic
t.good <- which(!is.na(fit2$F)) # filter out bad probes
qqt(fit2$t[t.good], fit2$df.total[t.good], main = "Moderated t statistic")

# volcano plot (log P-value vs log fold change)
colnames(fit2) # list contrast names
ct <- 1        # choose contrast of interest
volcanoplot(fit2, coef = ct, main = colnames(fit2)[ct], pch = 20,
            highlight = length(which(dT[, ct] != 0)), names = rep('+', nrow(fit2)))

# MD plot (log fold change vs mean log expression)
# highlight statistically significant (p-adj < 0.05) probes
plotMD(fit2, column = ct, status = dT[, ct], legend = F, pch = 20, cex = 1)
abline(h = 0)

# UMAP plot (dimensionality reduction)
gset <- na.omit(gset) # eliminate rows with NAs
gset <- gset[!duplicated(gset),]  # remove duplicates
ump <- umap(t(gset), n_neighbors = 5, random_state = 123)
par(mar = c(3, 3, 2, 6), xpd = TRUE)
plot(ump$layout, main = "UMAP plot, nbrs=5", xlab = "", ylab = "", col = gs, pch = 20, cex = 1.5)
legend("topright", inset = c(-0.15, 0), legend = levels(gs), pch = 20,
       col = 1:nlevels(gs), title = "Group", pt.cex = 1.5)

# mean-variance trend, helps to see if precision weights are needed
plotSA(fit2, main = paste("Mean variance trend,", toupper(file_name)))
