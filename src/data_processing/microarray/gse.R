# Differential expression analysis with limma
require(GEOquery)
require(limma)
require(umap)

working_dir <- file.path("R:/GeneAnalysis/data")
file_name <- "GSE8842"

load_object <- function(file) {
  tmp <- new.env()
  load(file = file, envir = tmp)
  tmp[[ls(tmp)[1]]]
}

# load series and platform data from GEO
gset <- load_object(file = file.path(working_dir, paste(file_name, ".rda", sep = "")))

# make proper column names to match toptable 
fvarLabels(gset) <- make.names(fvarLabels(gset))

# group membership for all samples
gsms <- as.numeric(gset@phenoData@data[["sample_type"]])
gsms <- paste0(gsms, collapse = "")
sml <- strsplit(gsms, split = "")[[1]]

# filter out excluded samples (marked as "X")
sel <- which(sml != "X")
sml <- sml[sel]
gset <- gset[, sel]

# log2 transformation
ex <- exprs(gset)
df <- as.data.frame(t(ex))
df$class <- sml
write.table(df, file = file.path(working_dir, paste(file_name, "_matrix.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)
remove(df)

qx <- as.numeric(quantile(ex, c(0., 0.25, 0.5, 0.75, 0.99, 1.0), na.rm = T))
LogC <- (qx[5] > 100) ||
  (qx[6] - qx[1] > 50 && qx[2] > 0)
if (LogC) { ex[which(ex <= 0)] <- NaN
  exprs(gset) <- log2(ex) }

# assign samples to groups and set up design matrix
gs <- factor(gset@phenoData@data[["sample_type"]])
groups <- levels(gs)
gset$group <- gs
design <- model.matrix(~group + 0, gset)
colnames(design) <- levels(gs)

fit <- lmFit(gset, design)  # fit linear model

# set up contrasts of interest and recalculate model coefficients
cts <- paste(groups[1], groups[2], sep = "-")
cont.matrix <- makeContrasts(contrasts = cts, levels = design)
fit2 <- contrasts.fit(fit, cont.matrix)

# compute statistics and table of top significant genes
fit2 <- eBayes(fit2, 0.01)
tT <- topTable(fit2, adjust = "fdr", sort.by = "B", number = 10000)
# tT <- subset(tT, select = c("ID", "adj.P.Val", "P.Value", "t", "B", "logFC", "Gene.symbol"))
write.table(tT, file = file.path(working_dir, paste(file_name, "_diff_features.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)

# Visualize and quality control test results.
# Build histogram of P-values for all genes. Normal test
# assumption is that most genes are not differentially expressed.
tT2 <- topTable(fit2, adjust = "fdr", sort.by = "B", number = Inf)
hist(tT2$adj.P.Val, col = "grey", border = "white", xlab = "P-adj",
     ylab = "Number of genes", main = "P-adj value distribution")

# summarize test results as "up", "down" or "not expressed"
dT <- decideTests(fit2, adjust.method = "fdr", p.value = 0.05)

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
# General expression data analysis
ex <- exprs(gset)

# box-and-whisker plot
dev.new(width = 3 + ncol(gset) / 6, height = 5)
ord <- order(gs)  # order samples by group
palette(c("#1B9E77", "#7570B3", "#E7298A", "#E6AB02", "#D95F02",
          "#66A61E", "#A6761D", "#B32424", "#B324B3", "#666666"))
par(mar = c(7, 4, 2, 1))
title <- paste(file_name, "/", annotation(gset), sep = "")
boxplot(ex[, ord], boxwex = 0.6, notch = T, main = title, outline = FALSE, las = 2, col = gs[ord])
legend("topleft", groups, fill = palette(), bty = "n")
dev.off()

# expression value distribution
par(mar = c(4, 4, 2, 1))
title <- paste(file_name, "/", annotation(gset), " value distribution", sep = "")
plotDensities(ex, group = gs, main = title, legend = "topright")

# UMAP plot (dimensionality reduction)
ex <- na.omit(ex) # eliminate rows with NAs
ex <- ex[!duplicated(ex),]  # remove duplicates
ump <- umap(t(ex), n_neighbors = 15, random_state = 123)
par(mar = c(3, 3, 2, 6), xpd = TRUE)
plot(ump$layout, main = "UMAP plot, nbrs=15", xlab = "", ylab = "", col = gs, pch = 20, cex = 1.5)
legend("topright", inset = c(-0.15, 0), legend = levels(gs), pch = 20,
       col = 1:nlevels(gs), title = "Group", pt.cex = 1.5)

# mean-variance trend, helps to see if precision weights are needed
plotSA(fit2, main = paste("Mean variance trend,", file_name))

