# Wang, Y.J., Schug, J., Won, K.J., Liu, C., Naji, A., Avrahami, D., Golson, M.L. and Kaestner, K.H., 2016. Single-cell transcriptomics of the human endocrine pancreas. Diabetes, 65(10), pp.3028-3038.

# Differential expression analysis with limma
require(limma)
require(umap)

working_dir <- file.path("R:/GeneAnalysis/data")
file_name <- "wang"
source("R:/GeneAnalysis/uhet/src/utility/create_sce.R")

### DATA
gset <- read.table(file.path(working_dir, paste(file_name, ".csv", sep = "")), header=T, stringsAsFactor=FALSE)
fDat <- gset[,1:7]
gset<-gset[,-1*c(1:7)]
rownames(gset) <- fDat$transcript
rownames(gset)[!is.na(fDat$gene)] <- fDat$gene[!is.na(fDat$gene)]
rownames(fDat) <- rownames(gset)

### ANNOTATIONS
ann1 <- read.delim(file.path(working_dir, paste(file_name, "_annotation.txt", sep = "")), sep="\n", header=F, stringsAsFactor=FALSE)
ANN1 <- ann1[c(38,48:49),]
ANN1 <- sapply(ANN1, function(y){strsplit(y, "\t")})
ann2 <- read.delim(file.path(working_dir, "wang2_annotation.txt"), sep="\n", header=F, stringsAsFactor=FALSE)
ANN2 <- ann2[c(38,48:49),]
ANN2 <- sapply(ANN2, function(y){strsplit(y, "\t")})
# Extract metadata
qualities <- list()
for (i in 1:length(ANN1)) {
        thing1 <- ANN1[[i]]
        thing1 <- thing1[-1]
        j <- 2
        if (i == 1) {j <- 1}
        thing1 <- matrix(unlist(strsplit(thing1, " ")), ncol=j, byrow=T)
        thing2 <- ANN2[[i]]
        thing2 <- thing2[-1]
        thing2 <- matrix(unlist(strsplit(thing2, " ")), ncol=j, byrow=T)
        qualities[[i]] = c(thing1[, j], thing2[, j])
}
classes <- data.frame(disease=qualities[[2]], cell_type1=qualities[[3]])
rownames(classes) <- paste("reads.", qualities[[1]], sep="")
colnames(gset) <- paste("reads.", qualities[[1]], sep="")
remove(ann1, ann2, ANN1, ANN2, qualities, fDat)

# Check cell ordering
classes$cell_type1 <- as.character(classes$cell_type1)
classes$cell_type1[classes$cell_type1 == "pp"] <- "gamma"
classes$cell_type1[classes$cell_type1 == "duct"] <- "ductal"

### SINGLECELLEXPERIMENT
gset <- create_sce_from_normcounts(gset, classes)
featureNames <- as.character(rownames(gset))
classes <- gset@colData@listData[["disease"]]
gset <- as.data.frame(t(gset@assays@data@listData[["logcounts"]]))

# group membership for all samples
# 0 : control       
# 1 : "T2D" and "T1D"
gsms <- c(0, 1, 1)
names(gsms) <- unique(classes)
gsms <- gsms[classes]
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# collect subtypes 
subtypes <- classes
write.table(as.data.frame(subtypes), file = file.path(working_dir, paste(file_name, "_types.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# log2 transformation
df <- gset
df$class <- sml
write.table(df, file = file.path(working_dir, paste(file_name, "_matrix.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)
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
write.table(tT, file = file.path(working_dir, paste(file_name, "_features.csv",
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

################################################################
# box-and-whisker plot
dev.new(width = 3 + ncol(gset) / 6, height = 5)
ord <- order(gs)  # order samples by group
palette(c("#1B9E77", "#7570B3", "#E7298A", "#E6AB02", "#D95F02",
          "#66A61E", "#A6761D", "#B32424", "#B324B3", "#666666"))
par(mar = c(7, 4, 2, 1))
title <- paste(toupper(file_name), sep = "")
boxplot(gset[, ord], boxwex = 0.6, notch = T, main = title, outline = FALSE, las = 2, col = gs[ord])
legend("topleft", groups, fill = palette(), bty = "n")
dev.off()

# expression value distribution
par(mar = c(4, 4, 2, 1))
title <- paste(toupper(file_name), " value distribution", sep = "")
plotDensities(gset, group = gs, main = title, legend = "topright")

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