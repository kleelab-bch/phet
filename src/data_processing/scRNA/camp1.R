# Camp, J.G., Sekine, K., Gerber, T., Loeffler-Wirth, H., Binder, H., Gac, M., Kanton, S., Kageyama, J., Damm, G., Seehofer, D. and Belicova, L., 2017. Multilineage communication regulates human liver bud development from pluripotency. Nature, 546(7659), pp.533-538.

# Differential expression analysis with limma
require(limma)
require(umap)
require(Matrix)
source("R:/GeneAnalysis/phet/src/data_processing/create_sce.R")
working_dir <- file.path("R:/GeneAnalysis/data")

file_name <- "camp1"

### DATA
# iPS = induced pluripotent stem cells (day 0)
# HE = hepatic endoderm (day 8)
# LB = liver bud = hepatic endoderm + supportive mesenchymal and endothelial cells
# MH = mature hepatocyte-like (day 21)
# DE = definitive endoderm (day 6)
# IH = immature hepatoblast-like (day 14)
# EC = endothelial cells
# HUVEC = human umbilical vein endothelial cells
# msc = mesenchymal stem cell

x1 <- read.delim(file.path(working_dir, paste(file_name, "_lineage.csv", sep = "")), sep = ",", stringsAsFactors = FALSE)
rownames(x1) <- x1[, 1]
experiment1 <- x1[, 2]
x1 <- x1[, -c(1, 2)]
x1 <- t(x1)

x2 <- read.delim(file.path(working_dir, paste(file_name, "_liverbud.csv", sep = "")), sep = ",", stringsAsFactors = FALSE)
rownames(x2) <- x2[, 1]
experiment2 <- x2[, 2]
assignment2 <- x2[, 3]
x2 <- x2[, -c(1, 2, 3)]
x2 <- t(x2)

### ANNOTATIONS
ann = read.table(file.path(working_dir, paste(file_name, "_annotation.txt", sep = "")), header = T, stringsAsFactors = FALSE)
tmp <- colnames(ann)
tmp <- matrix(unlist(strsplit(tmp, "_")), ncol = 2, byrow = T)
fixed_names <- paste(tmp[, 2], tmp[, 1], sep = "_")
colnames(ann) <- fixed_names
ann <- t(ann)
# Check duplicate cells
dups <- which(colnames(x1) %in% colnames(x2))
for (i in colnames(x1)[dups]) {
  if (sum(x2[, colnames(x2) == i] == x1[, colnames(x1) == i]) != 19020) {
    print(i);
  }
}

x1 <- x1[, -dups]
experiment1 <- experiment1[-dups];
gset <- cbind(x1, x2);

Stage <- c(experiment1, experiment2)
Type <- c(experiment1, assignment2);
Type[grepl("ih", Type)] <- "immature hepatoblast"
Type[grepl("mh", Type)] <- "mature hepatocyte"
Type[grepl("de", Type)] <- "definitive endoderm"
Type[grepl("EC", Type)] <- "endothelial"
Type[grepl("HE", Type)] <- "hepatic endoderm"
Type[grepl("MC", Type)] <- "mesenchymal stem cell"

Source1 <- rep("iPSC line TkDA3-4", times = length(experiment1))
Source2 <- rep("iPSC line TkDA3-4", times = length(experiment2));
Source2[experiment2 == "huvec"] = "HUVEC"
Source2[grepl("lb", experiment2)] = "liver bud"
Source2[grepl("msc", experiment2)] = "Mesenchymal stem cell"
Source <- c(Source1, Source2)
Age <- Stage
Age[Age == "de"] <- "6 days"
Age[Age == "ipsc"] <- "0 days"
Age[grepl("mh", Age)] <- "21 days"
Age[grepl("he", Age)] <- "8 days"
Age[grepl("lb", Age)] <- "liver bud"
Age[grepl("ih", Age)] <- "14 days"
Age[grepl("msc", Age)] <- "msc"
Batch <- Stage
Batch[grepl("lb1", Batch)] = "3"
Batch[grepl("lb2", Batch)] = "4"
Batch[grepl("lb3", Batch)] = "5"
Batch[grepl("lb4", Batch)] = "6"
Batch[grepl("lb5", Batch)] = "7"
Batch[grepl("lb6", Batch)] = "8"
Batch[grepl("1", Batch)] = "1"
Batch[grepl("2", Batch)] = "2"
Batch[grepl("de", Batch)] = "9"
Batch[grepl("ipsc", Batch)] = "10"
Batch[grepl("huvec", Batch)] = "11"
classes <- data.frame(Species = rep("Homo sapiens", times = length(Stage)), cell_type1 = Type, Source = Source, age = Age, batch = Batch)
rownames(classes) <- colnames(gset)

### SINGLECELLEXPERIMENT
gset <- create_sce_from_logcounts(gset, classes)
features <- as.character(rownames(gset))
selected.samples <- classes["Source"] == "iPSC line TkDA3-4"
classes <- gset@colData@listData[["cell_type1"]]
classes <- classes[selected.samples]
gset <- as.data.frame(t(gset@assays@data@listData[["logcounts"]]))
gset <- as.data.frame(gset[selected.samples,])

# group membership for all samples
# 0 (hepatic endoderm lineage): "definitive endoderm", "ipsc", "hepatic endoderm"
# 1 (post hepatic endoderm): "immature hepatoblast", "mature hepatocyte" 
gsms <- c(0, 1, 0, 1, 0)
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
temp <- tT[tT$adj.P.Val <= 0.01,]$ID
gset <- gset[temp,]
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