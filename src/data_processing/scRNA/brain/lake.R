# Lake, B.B., Ai, R., Kaeser, G.E., Salathia, N.S., Yung, Y.C., Liu, R., Wildberg, A., Gao, D., Fung, H.L., Chen, S. and Vijayaraghavan, R., 2016. Neuronal subtypes and diversity revealed by single-nucleus RNA sequencing of the human brain. Science, 352(6293), pp.1586-1590.

# Differential expression analysis with limma
require(limma)
require(umap)
require(Matrix)
source("R:/GeneAnalysis/phet/src/utility/create_sce.R")

working_dir <- file.path("R:/GeneAnalysis/data")
file_name <- "lake"

### DATA
#Prefrontal cortex - BA10
#Frontal cortex (eye fields) - BA8
#Auditory cortex (speech) - BA21
#Auditory cortex (Wernicke's area) - BA22
#Auditory cortex - BA41/42
#Visual cortex - BA17
# Problems: duplicate row names
# Human
x1 = read.delim(file.path(working_dir, paste(file_name, ".dat", sep = "")), "\t", header = F, stringsAsFactors = FALSE)
ann <- read.table(file.path(working_dir, paste(file_name, "_annotation.txt", sep = "")), header = T)
gene_names <- x1[, 1]
cell_names <- x1[1,]
cell_names <- as.character(unlist(cell_names))
cell_names <- cell_names[-1]
gene_names <- gene_names[-1]
x1 <- x1[-1, -1]
# not log-transformed tpms
exclude = duplicated(gene_names)
keep_cells = cell_names %in% ann[, 2]
x1 <- x1[, keep_cells]
cell_names <- cell_names[keep_cells]
colnames(x1) <- cell_names;
reorder <- order(colnames(x1))
x1 <- x1[, reorder]

### ANNOTATIONS
ann <- ann[ann[, 2] %in% colnames(x1),]
ann <- ann[order(ann[, 2]),]
SOURCE <- as.character(unlist(ann[, 4]))
BATCH <- as.character(unlist(ann[, 4]))
SOURCE[SOURCE == "BA10"] = "Prefrontal cortex"
SOURCE[SOURCE == "BA8"] = "Frontal cortex"
SOURCE[SOURCE == "BA21"] = "Auditory cortex (speech)"
SOURCE[SOURCE == "BA22"] = "Auditory cortex (Wernicke)"
SOURCE[SOURCE == "BA41"] = "Auditory cortex"
SOURCE[SOURCE == "BA17"] = "Visual cortex"
TYPE <- as.character(unlist(ann[3]))
AGE <- rep("51yo", times = length(BATCH))
stuff <- sub("_S.+", "", colnames(x1));
stuff <- matrix(unlist(strsplit(stuff, "_")), ncol = 2, byrow = T)
WELL <- stuff[, 2]
PLATE <- stuff[, 1]
gset <- apply(x1, 1, as.numeric)
gset <- t(gset)
colnames(gset) <- colnames(x1)
gset <- gset[!exclude,]
rownames(gset) <- gene_names[!exclude]
classes <- data.frame(Species = rep("Homo sapiens", times = length(TYPE)), cell_type1 = TYPE, Source = SOURCE, age = AGE, WellID = WELL, batch = BATCH, Plate = PLATE)
rownames(classes) <- colnames(x1)
remove(x1, ann, gene_names, stuff, AGE, BATCH, SOURCE, TYPE, WELL, PLATE, keep_cells, exclude, cell_names)

### SINGLECELLEXPERIMENT
gset <- create_sce_from_normcounts(gset, classes)
features <- as.character(rownames(gset))
classes <- gset@colData@listData[["Source"]]
gset <- as.data.frame(t(gset@assays@data@listData[["logcounts"]]))

# group membership for all samples
# 0 (non auditory cortex): "Frontal cortex", "Visual cortex", and "Prefrontal cortex"
# 1 (auditory cortex): "Auditory cortex (speech)", "Auditory cortex (Wernicke)", and "Auditory cortex"
gsms <- c(0, 0, 1, 1, 1, 0)
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