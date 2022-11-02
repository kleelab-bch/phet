# Pomeroy, S.L., Tamayo, P., Gaasenbeek, M., Sturla, L.M., Angelo, M., McLaughlin, M.E., Kim, J.Y., Goumnerova, L.C., Black, P.M., Lau, C. and Allen, J.C., 2002. Prediction of central nervous system embryonal tumour outcome based on gene expression. Nature, 415(6870), pp.436-442.

# Differential expression analysis with limma
require(limma)
require(umap)

working_dir <- file.path("R:/GeneAnalysis/data")
file_name <- "braintumor"

# load series and platform data from GEO
gset <- read.delim(file.path(working_dir, paste(file_name, ".tab", sep = "")),
                   header = TRUE, sep = "\t", check.names = FALSE,
                   stringsAsFactors = FALSE)
gset <- as.data.frame(as.matrix(gset)[3:nrow(gset),])
drop_cols <- c("", "sample", "class")
classes <- gset$class
featureNames <- colnames(gset)[!(names(gset) %in% drop_cols)]
gset <- gset[, featureNames]
gset <- as.data.frame(lapply(gset, as.numeric))

# group membership for all samples
# 0 (normal): normal cerebellum (mormal): 4 examples (10.0%)
# 1 (tumor): medulloblastoma (medulloblastoma), malignant glioma (glioma), Rhabdoid tumor (RhabdoidTu), and primitive neuroectodermal tumor (PNET): 36 examples (90.0%)
gsms <- c(1, 1, 1, 0, 1)
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
