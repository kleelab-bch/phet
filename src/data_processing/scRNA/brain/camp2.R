# Camp, J.G., Badsha, F., Florio, M., Kanton, S., Gerber, T., Wilsch-Br�uninger, M., Lewitus, E., Sykes, A., Hevers, W., Lancaster, M. and Knoblich, J.A., 2015. Human cerebral organoids recapitulate gene expression programs of fetal neocortex development. Proceedings of the National Academy of Sciences, 112(51), pp.15672-15677.

# Differential expression analysis with limma
require(limma)
require(umap)
require(Matrix)
source("R:/GeneAnalysis/phet/src/data_processing/create_sce.R")

working_dir <- file.path("R:/GeneAnalysis/data")
file_name <- "camp2"

### DATA
# Human cerebral organoids + fetal neocortex
# SMARTer
# Camp JG, Badsha F, Florio M, Kanton S et al. Human cerebral organoids recapitulate gene expression programs of fetal neocortex development. Proc Natl Acad Sci U S A 2015 Dec 22;112(51):15672-7. PMID: 26644564
# GSE75140
gset <- read.table(file.path(working_dir, paste(file_name, ".txt", sep = "")), header = T)
classes <- read.table(file.path(working_dir, paste(file_name, "_annotation.txt", sep = "")), header = T)

dnames <- as.character(gset[, 1]);
dnames <- strsplit(dnames, "_")
dnames <- lapply(dnames, function(x) { paste(sort(x), collapse = "_") })
dnames <- unlist(dnames)

rownames(gset) <- dnames;
gset <- gset[, -1]
gset <- gset[, -1 * length(gset[1,])]

### ANNOTATIONS
anames <- colnames(classes);
anames <- strsplit(anames, "_")
anames <- lapply(anames, function(x) { paste(sort(x[2:length(x)]), collapse = "_") })
anames <- unlist(anames)

anames[anames == "13wpc_F5_fetal"] = "12wpc_c1_F5_fetal"

colnames(classes) <- anames

gset <- t(gset)
gset <- gset[, order(colnames(gset))]
classes <- classes[, order(colnames(classes))]
classes <- t(classes)
# Using reported markers AUC > 0.85 from paper supplementary table

Neuron_Score <- colSums(gset[rownames(gset) %in% c("DCX", "MLLT11", "STMN2", "SOX4", "MYT1L"),]) - colSums(gset[rownames(gset) %in% c("ZFP36L1", "VIM"),])
is.neuron = Neuron_Score > 30

Mesenchyme_Score <- colSums(gset[rownames(gset) %in% c("COL3A1", "LUM", "S100A11", "DCN", "IFITM3", "COL1A2", "SPARC", "COL1A2", "COL1A1", "FTL", "ANXA2", "LGALS1", "MFAP4"),])
is.mesenchyme = Mesenchyme_Score > 90

dorsal_cortex_progenitors_score <- colSums(gset[rownames(gset) %in% c("C1orf61", "FABP7", "CREB5", "GLI3"),])
is.dcp <- dorsal_cortex_progenitors_score > 27

dorsal_cortex_neuron_score <- colSums(gset[rownames(gset) %in% c("NFIA", "NFIB", "ABRACL", "NEUROD6", "CAP2"),])
is.dcn <- dorsal_cortex_neuron_score > 25

ventral_progenitors_score <- colSums(gset[rownames(gset) %in% c("MTRNR2L12", "MDK", "BCAT1", "PRTG", "MGST1", "DLK1", "IGDCC3"),]) - colSums(gset[rownames(gset) %in% c("NFIB", "NFIA", "TUBB", "IFI44L", "PHLDA1", "CREB5"),])
is.vp <- ventral_progenitors_score > 20

RSPO_score <- colSums(gset[rownames(gset) %in% c("WLS", "TPBG"),])
# indistinct

Type = rep("Unknown", times = length(classes[, 1]))
Type[is.neuron] <- "neuron"
Type[is.mesenchyme] <- "mesenchyme"
Type[is.dcn] <- "dosal cortex neuron"
Type[is.dcp] <- "dosal cortex progenitor"
Type[is.vp] <- "ventral progenitor"

classes <- data.frame(Species = classes[, 2], cell_type1 = Type, Source = classes[, 4], age = classes[, 3], batch = classes[, 1])
rownames(classes) <- rownames(classes)

### SINGLECELLEXPERIMENT
gset <- create_sce_from_logcounts(gset, classes)
features <- as.character(rownames(gset))
classes <- gset@colData@listData[["Source"]]
gset <- as.data.frame(t(gset@assays@data@listData[["logcounts"]]))

# group membership for all samples
# 0 (fetal): Fetal neocortex
# 1 (non fetal): "Microdissected cortical-like ventricle from cerebral organoid" and "Dissociated whole cerebral organoid"
gsms <- c(0, 1, 1)
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