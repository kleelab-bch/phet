# Baron, M., Veres, A., Wolock, S.L., Faust, A.L., Gaujoux, R., Vetere, A., Ryu, J.H., Wagner, B.K., Shen-Orr, S.S., Klein, A.M. and Melton, D.A., 2016. A single-cell transcriptomic map of the human and mouse pancreas reveals inter-and intra-cell population structure. Cell systems, 3(4), pp.346-360.

# Differential expression analysis with limma
require(limma)
require(umap)
require(Matrix)
source("R:/GeneAnalysis/phet/src/data_processing/create_sce.R")
working_dir <- file.path("R:/GeneAnalysis/data")

file_name <- "baron"

### DATA
# human1
h1 <- read.csv(file.path(working_dir, paste(file_name, "1.csv", sep = "")), header = T)
rownames(h1) <- h1[, 1]
labels_h1 <- as.character(h1$assigned_cluster)
h1 <- h1[, 4:ncol(h1)]
h1 <- t(h1)
# human2
h2 <- read.csv(file.path(working_dir, paste(file_name, "2.csv", sep = "")), header = T)
rownames(h2) <- h2[, 1]
labels_h2 <- as.character(h2$assigned_cluster)
h2 <- h2[, 4:ncol(h2)]
h2 <- t(h2)
# human3
h3 <- read.csv(file.path(working_dir, paste(file_name, "3.csv", sep = "")), header = T)
rownames(h3) <- h3[, 1]
labels_h3 <- as.character(h3$assigned_cluster)
h3 <- h3[, 4:ncol(h3)]
h3 <- t(h3)
# human4
h4 <- read.csv(file.path(working_dir, paste(file_name, "4.csv", sep = "")), header = T)
rownames(h4) <- h4[, 1]
labels_h4 <- as.character(h4$assigned_cluster)
h4 <- h4[, 4:ncol(h4)]
h4 <- t(h4)

# merge data
gset <- cbind(h1, h2, h3, h4)
gset <- as.data.frame(gset)
remove(h1, h2, h3, h4)

### ANNOTATIONS
# human
classes <- data.frame(
  human = c(
    rep(1, length(labels_h1)),
    rep(2, length(labels_h2)),
    rep(3, length(labels_h3)),
    rep(4, length(labels_h4))
  ),
  cell_type1 = c(labels_h1, labels_h2, labels_h3, labels_h4))
rownames(classes) <- colnames(gset)
remove(labels_h1, labels_h2, labels_h3, labels_h4)

### SINGLECELLEXPERIMENT
gset <- create_sce_from_counts(gset, classes)
features <- as.character(rownames(gset))
classes <- gset@colData@listData[["cell_type1"]]
gset <- as.data.frame(t(gset@assays@data@listData[["logcounts"]]))

# group membership for all samples
# 0 (non ductal): "acinar", "beta", "delta", "activated_stellate", "alpha", "epsilon", "gamma", "endothelial", "quiescent_stellate", "macrophage", "schwann", "mast", and "t_cell"
# 1 (ductal) : "ductal"
gsms <- c(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
names(gsms) <- unique(classes)
gsms <- gsms[classes]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# collect subtypes 
subtypes <- classes
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
