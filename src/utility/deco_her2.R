require(deco, quietly = TRUE)
require(dplyr, quietly = TRUE)

working_dir <- file.path("R:/GeneAnalysis/data")

file_name <- "her2"
nBatches <- 1000
subsampleSize <- 10
iterations <- 1000
q.val <- 0.01

# load positive and negative HER2 data
X_control <- read.csv(file.path(working_dir, paste(file_name, "_negative_matrix.csv", sep = "")), header = T)
X_control <- as.data.frame(t(X_control))
X_control <- data.matrix(X_control)
X_control <- X_control[!rowSums(!is.finite(X_control)),]
featureIDs <- seq(0, nrow(X_control) - 1)
rownames(X_control) <- featureIDs
X_control <- SummarizedExperiment(assays = list(counts = X_control))

X_case <- read.csv(file.path(working_dir, paste(file_name, "_positive_matrix.csv", sep = "")), header = T)
X_case <- as.data.frame(t(X_case))
X_case <- data.matrix(X_case)
X_case <- X_case[!rowSums(!is.finite(X_case)),]
if (length(featureIDs) != nrow(X_case)) {
  stop("Feature size for both datasets are not same!")
}
rownames(X_case) <- featureIDs
X_case <- SummarizedExperiment(assays = list(counts = X_case))

#######################################################################
# RUNNING SUBSAMPLING OF DATA: BINARY design (two classes of samples) #
#######################################################################

pb <- txtProgressBar(min = 0,      # Minimum value of the progress bar
                     max = nBatches, # Maximum value of the progress bar
                     style = 3,    # Progress bar style (also available style = 1 and style = 2)
                     width = 80,   # Progress bar width. Defaults to getOption("width")
                     char = "=")   # Character used to create the bar
batch_matrix <- matrix(0, nrow = length(featureIDs), ncol = nBatches)
for (batch_idx in 1:nBatches) {
  temp <- sample(ncol(assay(X_case)), size = subsampleSize, replace = FALSE, prob = NULL)
  gset <- cbind(assay(X_control), assay(X_case)[, temp])
  classes <- rep(c(0, 1), c(ncol(assay(X_control)), subsampleSize))
  classes <- as.integer(classes)
  names(classes) <- colnames(gset)
  gset <- SummarizedExperiment(assays = list(counts = gset))
  subSampling <- suppressMessages(decoRDA(data = assay(gset), classes = classes,
                                          q.val = q.val, rm.xy = FALSE, r = NULL,
                                          control = "0", annot = FALSE,
                                          iterations = iterations,
                                          bpparam = MulticoreParam()))
  remove(gset)
  subSampling <- subSampling[["subStatFeature"]]["ID"]
  subSampling <- as.numeric(unlist(subSampling))
  feature_order <- 1
  for (variable in subSampling) {
    batch_matrix[variable + 1, batch_idx] = feature_order
    feature_order <- feature_order + 1
  }
  setTxtProgressBar(pb, batch_idx)
}
close(pb) # Close the connection
colnames(batch_matrix) <- seq(0, nBatches - 1)
write.table(batch_matrix,
            file = file.path(working_dir, paste(file_name, "_deco.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE, col.names = FALSE)
