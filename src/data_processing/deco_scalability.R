require(deco, quietly = TRUE)
require(dplyr, quietly = TRUE)
require(SummarizedExperiment)
require(Matrix)
require(BiocParallel) # for parallel computation

# Computing in shared memory
bpparam <- MulticoreParam()

working_dir <- file.path("R:/GeneAnalysis/data")
result_dir <- file.path("R:/GeneAnalysis/result/")

# srbct, lung, baron, and pulseseq
file_name <- "srbct"
q.val <- 0.01
num_iterations <- 10

# load data
gset <- readMM(file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))
gset <- as.data.frame(data.matrix(gset))
gset <- data.frame(t(gset))
gset <- data.matrix(gset)
gset <- gset[!rowSums(!is.finite(gset)),]
featureIDs <- seq(0, nrow(gset) - 1)
rownames(gset) <- featureIDs
classes <- read.csv(file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
                    header = T)
classes <- classes$classes
names(classes) <- colnames(gset)
gset <- SummarizedExperiment(assays = list(counts = gset))

#######################################################################
# RUNNING SUBSAMPLING OF DATA: BINARY design (two classes of samples) #
#######################################################################

times_matrix <- matrix(0, nrow = num_iterations, ncol = 1)
pb <- txtProgressBar(min = 0,      # Minimum value of the progress bar
                     max = num_iterations, # Maximum value of the progress bar
                     style = 3,    # Progress bar style (also available style = 1 and style = 2)
                     width = 80,   # Progress bar width. Defaults to getOption("width")
                     char = "=")   # Character used to create the bar
for (time_idx in 1:num_iterations) {
  currentTime <- Sys.time()
  subSampling <- decoRDA(data = assay(gset), classes = classes, q.val = q.val,
                         rm.xy = FALSE, r = NULL, control = "0", annot = FALSE,
                         iterations = 1000, bpparam = bpparam)
  times_matrix[time_idx] = Sys.time() - currentTime
  setTxtProgressBar(pb, time_idx)
  remove(subSampling)
}
close(pb) # Close the connection
remove(subSampling, gset)
write.table(data.frame("Times" = times_matrix),
            file = file.path(result_dir, paste(file_name, "_deco_times.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)