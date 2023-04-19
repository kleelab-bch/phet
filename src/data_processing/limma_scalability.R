require(limma)
require(dplyr, quietly = TRUE)
require(Matrix)

working_dir <- file.path("R:/GeneAnalysis/data")
result_dir <- file.path("R:/GeneAnalysis/result/")

# srbct, lung, baron, and pulseseq
file_name <- "baron"
num_iterations <- 10

# load data
gset <- readMM(file.path(working_dir, paste(file_name, "_matrix.mtx", sep = "")))
gset <- as.data.frame(data.matrix(gset))
classes <- read.csv(file.path(working_dir, paste(file_name, "_classes.csv", sep = "")),
                    header = T)
classes <- classes$classes
names(classes) <- rownames(gset)
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
featureIDs <- seq(0, nrow(gset) - 1)
rownames(gset) <- featureIDs

### LIMMA
times_matrix <- matrix(0, nrow = num_iterations, ncol = 1)
pb <- txtProgressBar(min = 0,      # Minimum value of the progress bar
                     max = num_iterations, # Maximum value of the progress bar
                     style = 3,    # Progress bar style (also available style = 1 and style = 2)
                     width = 80,   # Progress bar width. Defaults to getOption("width")
                     char = "=")   # Character used to create the bar
for (time_idx in 1:num_iterations) {
  currentTime <- Sys.time()
  fit <- lmFit(gset, design)  # fit linear model
  cts <- paste(groups[1], groups[2], sep = "-")
  cont.matrix <- makeContrasts(contrasts = cts, levels = design)
  fit2 <- contrasts.fit(fit, cont.matrix)
  fit2 <- eBayes(fit2, 0.01)
  times_matrix[time_idx] = Sys.time() - currentTime
  setTxtProgressBar(pb, time_idx)
  remove(fit, cts, cont.matrix, fit2)
}
close(pb) # Close the connection
remove(gset)

write.table(data.frame("Times" = times_matrix),
            file = file.path(result_dir, paste(file_name, "_limma_times.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)