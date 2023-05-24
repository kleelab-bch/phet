library(miloR)
library(SingleCellExperiment)
library(scater)
library(dplyr)
library(patchwork)
working_dir <- file.path("R:/GeneAnalysis/result/plasschaert_mouse")

# Load data
mtecs_colData <- read.csv(file.path(working_dir,
                                    paste("plasschaert_mouse_groups.csv",
                                          sep = "")),
                          header = FALSE, sep = ",", check.names = FALSE,
                          stringsAsFactors = FALSE)
mtecs_colData <- as.data.frame(t(mtecs_colData))
colnames(mtecs_colData) <- mtecs_colData[1,]
rownames(mtecs_colData) <- NULL
mtecs_colData <- mtecs_colData[2:dim(mtecs_colData)[1],]
mtecs_colData$stages <- mtecs_colData$timepoints
mtecs_colData$stages[mtecs_colData$stages != "uninjured"] = "injured"

mtecs_features <- read.csv(file.path(working_dir,
                                     paste("plasschaert_mouse_phet_b_features.csv",
                                           sep = "")),
                           header = FALSE, sep = ",", check.names = FALSE,
                           stringsAsFactors = FALSE)
mtecs_data <- read.csv(file.path(working_dir,
                                 paste("plasschaert_mouse_phet_b_expression.csv", sep = "")),
                       header = FALSE, sep = ",", check.names = FALSE,
                       stringsAsFactors = FALSE)
colnames(mtecs_data) <- mtecs_features$V1
mtecs_data <- SingleCellExperiment(t(mtecs_data), colData = mtecs_colData)
mtecs_data@assays@data@listData[["logcounts"]] <- mtecs_data@assays@data@listData[[1]]
mtecs_data@assays@data@listData[[1]] <- NULL
mtecs_data <- runPCA(mtecs_data, ncomponents = 50)
mtecs_data <- runUMAP(mtecs_data)
# Visualize the data
plotReducedDim(mtecs_data, colour_by = "timepoints", dimred = "UMAP")

# Create a Milo object
mtecs_milo <- Milo(mtecs_data)
# 1. Construct KNN graph
mtecs_milo <- buildGraph(mtecs_milo, k = 30, d = 30, reduced.dim = "PCA")
# 2. Defining representative neighbourhoods on the KNN graph
mtecs_milo <- makeNhoods(mtecs_milo, prop = 0.1, k = 30, d = 30,
                         refined = TRUE, reduced_dims = "PCA")
plotNhoodSizeHist(mtecs_milo)
# 3. Counting cells in neighbourhoods
mtecs_milo <- countCells(mtecs_milo,
                         meta.data = data.frame(colData(mtecs_milo)),
                         sample = "samples")
# 3.1 Computing neighbourhood connectivity
mtecs_milo <- calcNhoodDistance(mtecs_milo, d = 30, reduced.dim = "PCA")
# 3.2 Defining experimental design
mtecs_design <- data.frame(colData(mtecs_milo))[, c("samples",
                                                    "timepoints", "stages")]
mtecs_design <- distinct(mtecs_design)
rownames(mtecs_design) <- mtecs_design$samples
# 4. Differential abundance testing
da_results <- testNhoods(mtecs_milo, design = ~stages,
                         design.df = mtecs_design)
da_results %>%
  arrange(SpatialFDR) %>%
  head()
#4.1 Inspecting DA testing results
ggplot(da_results, aes(PValue)) + geom_histogram(bins = 50)
ggplot(da_results, aes(logFC, -log10(SpatialFDR))) +
  geom_point() +
  geom_hline(yintercept = 1) ## Mark significance threshold (10% FDR)
## Plot single-cell UMAP
mtecs_milo <- buildNhoodGraph(mtecs_milo)
UMAP_pl <- plotReducedDim(mtecs_milo, dimred = "UMAP", colour_by = "subtypes",
                          text_by = "subtypes", text_size = 3) +
  guides(fill = "none")
## Plot neighbourhood graph
nh_graph_pl <- plotNhoodGraphDA(mtecs_milo, da_results, layout = "UMAP",
                                alpha = 0.05)
UMAP_pl +
  nh_graph_pl +
  plot_layout(guides = "collect")

da_results <- annotateNhoods(mtecs_milo, da_results, coldata_col = "subtypes")
ggplot(da_results, aes(subtypes_fraction)) +
  geom_histogram(bins = 50)
plotDAbeeswarm(da_results, group.by = "subtypes")