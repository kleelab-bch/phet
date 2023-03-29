require(deco, quietly = TRUE)
require(dplyr, quietly = TRUE)
require(BiocParallel) # for parallel computation

# Computing in shared memory
bpparam <- MulticoreParam()

working_dir <- file.path("R:/GeneAnalysis/data")

file_name <- "simulated_normal"
outlier_per <- paste("0", seq(9), "_", sep = "")
data_type <- c("minority_features", "mixed_features")
iterations <- 1000
q.val <- 0.01

for (per in outlier_per) {
  for (t in data_type) {
    print(paste("Loading ", file_name, per, t, ".csv ...", sep = ""))
    # Load data
    gset <- read.csv(file.path(working_dir, 
                               paste(file_name, per, t, ".csv", sep = "")), 
                     header = T)
    classes <- gset$class
    gset <- gset[!(names(gset) %in% c("class"))]
    gset <- as.data.frame(t(gset))
    names(classes) <- colnames(gset)
    gset <- data.matrix(gset)
    gset <- gset[!rowSums(!is.finite(gset)),]
    colnames(gset) <- names(classes)
    featureIDs <- seq(0, nrow(gset) - 1)
    rownames(gset) <- featureIDs
    gset <- SummarizedExperiment(assays = list(counts = gset))
    
    #######################################################################
    # RUNNING SUBSAMPLING OF DATA: BINARY design (two classes of samples) #
    #######################################################################
    
    subSampling <- decoRDA(data = assay(gset), classes = classes, q.val = q.val,
                           rm.xy = FALSE, r = NULL, control = "0", annot = FALSE,
                           iterations = iterations, bpparam = bpparam)
    StatFeature <- subSampling[["subStatFeature"]]
    StatFeature <- StatFeature[c("ID", "Standard.Chi.Square")]
    colnames(StatFeature) <- c("features", "score")
    write.table(as.data.frame(StatFeature),
                file = file.path(working_dir, 
                                 paste(file_name, per, t, "_deco.csv", sep = "")), 
                sep = ",", quote = FALSE, row.names = FALSE)
    remove(StatFeature, gset)
  }
}