# Grubman, A., Chew, G., Ouyang, J.F., Sun, G., Choo, X.Y., McLean, C., Simmons, R.K., Buckberry, S., Vargas-Landin, D.B., Poppe, D. and Pflueger, J., 2019. A single-cell atlas of entorhinal cortex from individuals with Alzheimer's disease reveals cell-type-specific gene expression regulation. Nature neuroscience, 22(12), pp.2087-2097.

# Differential expression analysis with limma
require(limma)
require(umap)
require(Seurat)
require(Matrix)
source("R:/GeneAnalysis/phet/src/data_processing/create_sce.R")
working_dir <- file.path("R:/GeneAnalysis/data")

file_name <- "chew"

### Load data
# metadata
classes <- read.table(file.path(working_dir,
                                 paste("GSE138852_covariates.csv", sep = "")),
                       header = TRUE, sep = ",", row.names = 1,
                       check.names = FALSE, stringsAsFactors = FALSE)
rownames(classes) <- NULL
colnames(classes) <- c("cell_type1", "Subtypes", "cellType_batchCond", 
                       "subclustID", "subclustCond")
gset <- read.table(file.path(working_dir,
                             paste("GSE138852_counts.csv", sep = "")),
                   header = TRUE, sep = ",", row.names = 1,
                   check.names = FALSE, stringsAsFactors = FALSE)

### SINGLECELLEXPERIMENT
gset <- create_sce_from_counts(gset, classes)
featureNames <- as.character(rownames(gset))
classes <- gset@colData@listData[["cell_type1"]]
subtypes <- gset@colData@listData[["Subtypes"]]
gset <- as.data.frame(t(gset@assays@data@listData[["logcounts"]]))

# group membership for all samples
# 0 (CT cells): Control
# 1 (AD cells): Alzheimer's disease (AD)
gsms <- c(1, 0)
names(gsms) <- unique(classes)
gsms <- gsms[classes]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# save subtypes 
write.table(as.data.frame(subtypes), 
            file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# save features data
gset$class <- sml
write.table(gset, 
            file = file.path(working_dir, paste(file_name, "_matrix.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)
gset$class <- NULL

#######################################################################
################## Differential Expression Analysis ###################
#######################################################################
# Assign samples to groups and set up design matrix
gs <- factor(sml)
groups <- make.names(c("Control", "Case"))
levels(gs) <- groups
gset$group <- gs
design <- model.matrix(~group + 0, gset)
colnames(design) <- levels(gs)
gset <- gset[, !(names(gset) %in% "group")]
gset <- t(gset)
gset[is.na(gset)] <- 0

### LIMMA
fit <- lmFit(gset, design)  # fit linear model
# set up contrasts of interest and recalculate model coefficients
cts <- paste(groups[1], groups[2], sep = "-")
cont.matrix <- makeContrasts(contrasts = cts, levels = design)
fit2 <- contrasts.fit(fit, cont.matrix)
# compute statistics and table of top significant genes
fit2 <- eBayes(fit2, 0.01)
tT <- topTable(fit2, adjust = "fdr", sort.by = "B", number = 100000)
temp <- rownames(tT)
rownames(tT) <- NULL
tT <- cbind("ID" = temp, tT)
write.table(tT, file = file.path(working_dir, paste(file_name, "_limma_features.csv",
                                                    sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

#######################################################################
#################### Visualization of LIMMA Results ###################
#######################################################################
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

# expression value distribution
par(mar = c(4, 4, 2, 1))
title <- paste(toupper(file_name), " value distribution", sep = "")
plotDensities(gset, group = gs, main = title, legend = "topright")

# UMAP plot (dimensionality reduction)
gset <- na.omit(gset) # eliminate rows with NAs
gset <- gset[!duplicated(gset),]  # remove duplicates
temp <- tT[tT$adj.P.Val <= 0.01, ]$ID
gset <- gset[temp, ]
classes <- factor(classes)
ump <- umap(t(gset), n_neighbors = 5, min_dist = 0.01, n_epochs = 2000, 
            random_state = 123)
par(mar = c(3, 3, 2, 6), xpd = TRUE)
plot(ump$layout, main = paste(toupper(file_name), "\nFeatures: ", length(temp)), 
     xlab = "", ylab = "", 
     col = classes, pch = 20, cex = 1.5)
legend("topright", inset = c(-0.15, 0), legend = levels(classes), pch = 20,
       col = 1:nlevels(classes), title = "Group", pt.cex = 1.5)

# mean-variance trend, helps to see if precision weights are needed
plotSA(fit2, main = paste("Mean variance trend,", toupper(file_name)))