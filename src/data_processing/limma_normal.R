require(limma)
require(dplyr, quietly = TRUE)

working_dir <- file.path("R:/GeneAnalysis/data")

file_name <- "simulated_normal"
outlier_per <- paste("0", seq(9), "_", sep = "")
outlier_per[length(outlier_per) + 1] <- "10_"
data_type <- c("minority_features", "mixed_features")

for (per in outlier_per) {
  for (t in data_type) {
    print(paste("Loading ", file_name, per, t, ".csv...", sep = ""))
    # Load data
    gset <- read.csv(file.path(working_dir,
                               paste(file_name, per, t, ".csv", sep = "")),
                     header = T)
    classes <- gset$class
    gset <- gset[!(names(gset) %in% c("class"))]
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
    write.table(tT, file = file.path(working_dir,
                                     paste(file_name, per, t, "_limma_features.csv",
                                           sep = "")),
                sep = ",", quote = FALSE, row.names = FALSE)
  }
}