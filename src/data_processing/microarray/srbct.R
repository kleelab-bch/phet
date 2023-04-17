# Khan, J., Wei, J.S., Ringner, M., Saal, L.H., Ladanyi, M., Westermann, F., Berthold, F., Schwab, M., Antonescu, C.R., Peterson, C. and Meltzer, P.S., 2001. Classification and diagnostic prediction of cancers using gene expression profiling and artificial neural networks. Nature medicine, 7(6), pp.673-679.

# Differential expression analysis with limma
require(limma)
require(edgeR)
require(dearseq)
require(matrixTests)
require(umap)
require(Matrix)

working_dir <- file.path("R:/GeneAnalysis/data")
file_name <- "srbct"

# load series and platform data from GEO
gset <- read.delim(file.path(working_dir, paste(file_name, ".tab", sep = "")),
                   header = TRUE, sep = "\t", check.names = FALSE,
                   stringsAsFactors = FALSE)
gset <- as.data.frame(as.matrix(gset)[3:nrow(gset),])
drop_cols <- c("", "sample", "class")
classes <- gset$class
features <- colnames(gset)[!(names(gset) %in% drop_cols)]
gset <- gset[, features]
gset <- as.data.frame(lapply(gset, as.numeric))
features <- colnames(gset)[!(names(gset) %in% drop_cols)]

# group membership for all samples
# 0: Ewing's sarcoma (EWS) and rhabdomyosarcoma (RMS): 54 examples 
# 1: Burkitt's lymphoma (BL) and neuroblastoma (NB): 29 examples
gsms <- c(0, 1, 1, 0)
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

### edgeR
count_norm <- DGEList(counts=gset, group=gs)
count_norm <- calcNormFactors(count_norm)
edger_design <- model.matrix(~gs)
count_norm <- estimateDisp(count_norm, edger_design) 
# perform quasi-likelihood F-tests:
fit <- glmQLFit(count_norm, edger_design)
fit <- glmQLFTest(fit, coef=2)
tT <- topTags(fit, n=nrow(gset), adjust.method = "BH", sort.by = "PValue", 
              p.value = 1)
tT <- tT[["table"]]
tT <- as.data.frame(cbind("ID"= rownames(tT), "logFC" = tT$logFC, "F" = tT$F, 
                          "FDR" = tT$FDR, "adj.P.Val" = tT$PValue))
write.table(tT, file = file.path(working_dir, paste(file_name, "_edger_features.csv",
                                                    sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)
remove(count_norm, edger_design, fit, tT)

### dearseq
count_norm <- DGEList(counts=gset, group=gs)
# perform TMM normalization and transfer to CPM (Counts Per Million)
count_norm <- calcNormFactors(count_norm, method="TMM")
count_norm <- cpm(count_norm, log=TRUE)
tT <- dear_seq(exprmat=count_norm, 
               variables2test=matrix(as.numeric(sml), ncol=1), 
               which_test="asymptotic", padjust_methods="BH",
               parallel_comp=F, preprocessed=T)
tT<-tT[["pvals"]]
tT$adjPval <- as.numeric(as.character(tT$adjPval))
tT <- tT[order(tT$adjPval, decreasing = FALSE), ]
tT <- as.data.frame(cbind("ID"= rownames(tT), "P.Value" = tT$rawPval, 
                          "adj.P.Val" = tT$adjPval))
write.table(tT, file = file.path(working_dir, paste(file_name, "_dearseq_features.csv",
                                                    sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)
remove(count_norm, tT)

### Wilcoxon rank-sum test
count_norm <- DGEList(counts=gset, group=gs)
# perform TMM normalization and transfer to CPM (Counts Per Million)
count_norm <- calcNormFactors(count_norm, method="TMM")
count_norm <- cpm(count_norm)
count_norm <- as.data.frame(count_norm)
dataMem1 <- count_norm[, c(which(gs == levels(gs)[1]))]
dataMem2 <- count_norm[, c(which(gs == levels(gs)[2]))]
tT <- row_wilcoxon_twosample(dataMem1, dataMem2)
fdr <- p.adjust(tT$pvalue, method = "BH")
tT <- as.data.frame(cbind("ID"= rownames(tT), "statistic" = tT$statistic, 
                          "P.Value" = tT$pvalue, "adj.P.Val" = fdr))
tT$adj.P.Val <- as.numeric(as.character(tT$adj.P.Val))
tT <- tT[order(tT$adj.P.Val, decreasing = FALSE), ]
write.table(tT, file = file.path(working_dir, paste(file_name, "_wilcoxon_features.csv",
                                                    sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)
remove(count_norm, dataMem1, dataMem2, tT, fdr)

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