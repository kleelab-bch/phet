require(limma)
require(dplyr, quietly = TRUE)

working_dir <- file.path("R:/GeneAnalysis/data")

file_name <- "her2"
num_batches <- 1000
subsampleSize <- 10
p.value <- 0.01

# load positive and negative HER2 data
X_control <- read.csv(file.path(working_dir, 
                                paste(file_name, "_negative_matrix.csv", sep = "")), 
                      header = T)
X_control <- as.data.frame(t(X_control))
X_control <- data.matrix(X_control)
X_control <- X_control[!rowSums(!is.finite(X_control)),]
featureIDs <- seq(0, nrow(X_control) - 1)
rownames(X_control) <- featureIDs

X_case <- read.csv(file.path(working_dir, 
                             paste(file_name, "_positive_matrix.csv", sep = "")), 
                   header = T)
X_case <- as.data.frame(t(X_case))
X_case <- data.matrix(X_case)
X_case <- X_case[!rowSums(!is.finite(X_case)),]
if (length(featureIDs) != nrow(X_case)) {
  stop("Feature size for both datasets are not same!")
}
rownames(X_case) <- featureIDs

#######################################################################
############### LIMMA: Differential Expression Analysis ###############
#######################################################################
pb <- txtProgressBar(min = 0,      # Minimum value of the progress bar
                     max = num_batches, # Maximum value of the progress bar
                     style = 3,    # Progress bar style (also available style = 1 and style = 2)
                     width = 80,   # Progress bar width. Defaults to getOption("width")
                     char = "=")   # Character used to create the bar
limma_matrix <- matrix(0, nrow = length(featureIDs), ncol = num_batches)
limma_distr_matrix <- matrix(0, nrow = length(featureIDs), ncol = num_batches)
for (batch_idx in 1:num_batches) {
  temp <- sample(ncol(X_case), size = subsampleSize, replace = FALSE, prob = NULL)
  gset <- as.data.frame(t(cbind(X_control, X_case[, temp])))
  classes <- rep(c(0, 1), c(ncol(X_control), subsampleSize))
  # Assign samples to groups and set up design matrix
  gs <- factor(classes)
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
  tT <- topTable(fit2, adjust = "fdr", sort.by = "B", p.value = p.value)
  tT <- as.integer(rownames(tT))
  
  tT_distr <- topTable(fit2, adjust = "fdr", sort.by = "none", number = 100000)
  rownames(tT_distr) <- NULL
  remove(temp, gset, classes, groups, gs, design, fit, cts, cont.matrix, fit2)
  
  feature_order <- 1
  for (variable in tT) {
    limma_matrix[variable + 1, batch_idx] <- feature_order
    feature_order <- feature_order + 1
  }
  
  limma_distr_matrix[, batch_idx] <- tT_distr$B
  remove(tT, tT_distr)
  setTxtProgressBar(pb, batch_idx)
}
close(pb) # Close the connection

colnames(limma_matrix) <- seq(0, num_batches - 1)
write.table(limma_matrix,
            file = file.path(working_dir, 
                             paste(file_name, "_limma_features.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE, col.names = FALSE)
colnames(limma_distr_matrix) <- seq(0, num_batches - 1)
write.table(limma_distr_matrix,
            file = file.path(working_dir, 
                             paste(file_name, "_limma_distr_features.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE, col.names = FALSE)
