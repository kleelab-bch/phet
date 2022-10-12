require(deco)
require(dplyr)

working_dir <- file.path("R:/GeneAnalysis/data")

# Simulated Datasets:
# 1. simulated_normal, simulated_normal_minority, simulated_normal_minority_features,
# simulated_normal_mixed, simulated_normal_mixed_features
# 2. simulated_weak, simulated_weak_minority, simulated_weak_minority_features,
# simulated_weak_mixed, simulated_weak_mixed_features

# Micro-array datasets:
# allgse412, amlgse2191, bc_ccgse3726, bcca1, bcgse349_350, bladdergse89, braintumor,
# cmlgse2535, colon, dlbcl, ewsgse967, gastricgse2685, glioblastoma, leukemia_golub,
# ll_gse1577_2razreda, lung, lunggse1987, meduloblastomigse468, mll, myelodysplastic_mds1,
# myelodysplastic_mds2, pdac, prostate, prostategse2443, srbct, and tnbc

# scRNA datasets:
# camp2, darmanis, lake, yan, camp1, baron, segerstolpe, wang, li, and patel

file_name <- "lake"
iterations <- 1000
q.val <- 0.01

# load data
gset <- read.csv(file.path(working_dir, paste(file_name, "_matrix.csv", sep = "")), header = T)
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
                       iterations = iterations, bpparam = MulticoreParam())
StatFeature <- subSampling[["subStatFeature"]]
StatFeature <- StatFeature[c("ID", "Standard.Chi.Square")]
colnames(StatFeature) <- c("features", "score")
write.table(as.data.frame(StatFeature),
            file = file.path(working_dir, paste(file_name, "_deco.csv", sep = "")),
            sep = ",", quote = FALSE, row.names = FALSE)
remove(StatFeature, gset)

#########################################################################################
# RUNNING NSCA STEP: Looking for subclasses within a category/class of samples compared #
#########################################################################################

# subClasses <- decoNSCA(sub = subSampling, v = 80, method = "ward.D", 
#                        bpparam = MulticoreParam(), k.control = 2, k.case = 2, 
#                        samp.perc = 0.05, rep.thr = 3)

########################################################
# PDF report with feature-sample patterns or subgroups #
########################################################

# working_dir <- file.path("R:/GeneAnalysis/result")
# path = file.path(working_dir, paste(file_name, "_deco.pdf", sep = ""))
# decoReport(subClasses, subSampling, pdf.file = path, cex.names = 0.3,
#            print.annot = TRUE)
