library(SingleCellExperiment)
library(scater)

create_sce_from_counts <- function(counts, colData, rowData = NULL) {
  sceset <- SingleCellExperiment(assays = list(counts = as.matrix(counts)),
                                 colData = colData, rowData = rowData)
  # this function writes to logcounts slot
  exprs(sceset) <- log2(calculateCPM(sceset, size.factors = NULL) + 1)
  # use gene names as feature symbols
  rowData(sceset)$feature_symbol <- rownames(sceset)
  # remove features with duplicated names
  if (is.null(rowData)) {
    sceset <- sceset[!duplicated(rowData(sceset)$feature_symbol),]
  }
  # QC
  is.spike <- grepl("^ERCC-", rownames(sceset))
  sceset <- splitAltExps(sceset, ifelse(is.spike, "ERCC", "gene"))
  return(sceset)
}

create_sce_from_normcounts <- function(normalizedcounts, colData, rowData = NULL) {
  sceset <- SingleCellExperiment(assays = list(normcounts = as.matrix(normalizedcounts)),
                                 colData = colData,
                                 rowData = rowData)
  logcounts(sceset) <- log2(normcounts(sceset) + 1)
  # use gene names as feature symbols
  rowData(sceset)$feature_symbol <- rownames(sceset)
  # remove features with duplicated names
  if (is.null(rowData)) {
    sceset <- sceset[!duplicated(rowData(sceset)$feature_symbol),]
  }
  # QC
  is.spike <- grepl("^ERCC-", rownames(sceset))
  sceset <- splitAltExps(sceset, ifelse(is.spike, "ERCC", "gene"))
  return(sceset)
}

create_sce_from_logcounts <- function(logcounts, colData, rowData = NULL) {
  sceset <- SingleCellExperiment(assays = list(logcounts = as.matrix(logcounts)),
                                 colData = colData,
                                 rowData = rowData)
  # use gene names as feature symbols
  rowData(sceset)$feature_symbol <- rownames(sceset)
  # remove features with duplicated names
  if (is.null(rowData)) {
    sceset <- sceset[!duplicated(rowData(sceset)$feature_symbol),]
  }
  # QC
  is.spike <- grepl("^ERCC-", rownames(sceset))
  sceset <- splitAltExps(sceset, ifelse(is.spike, "ERCC", "gene"))
  return(sceset)
}
