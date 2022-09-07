require(deco)

data(ALCLdata)

## Classes vector to run a supervised analysis to compare both classes.
classes.ALCL <- colData(ALCL)[, "Alk.positivity"]
names(classes.ALCL) <- colnames(ALCL)

#######################################################################
# RUNNING SUBSAMPLING OF DATA: BINARY design (two classes of samples) #
#######################################################################

sub.ma.3r.1K <- decoRDA(data = assay(ALCL), classes = classes.ALCL, q.val = 0.01,
                        rm.xy = TRUE, r = NULL, control = "pos", annot = FALSE,
                        id.type = "ENSEMBL", iterations = 5, pack.db = "Homo.sapiens")


#########################################################################################
# RUNNING NSCA STEP: Looking for subclasses within a category/class of samples compared #
#########################################################################################
deco.results.ma <- decoNSCA(sub = sub.ma.3r.1K, v = 80, method = "ward.D", bpparam = bpparam,
                            k.control = 3, k.case = 3, samp.perc = 0.05, rep.thr = 10)


# Phenotypical data from TCGA RNAseq samples.
colData(ALCL)

########################################################
# PDF report with feature-sample patterns or subgroups #
########################################################
## Generate PDF report with relevant information and several plots.

## Binary example (ALK+ vs ALK-) -not run as example-
decoReport(deco.results.ma, sub.ma.3r.1K,
           pdf.file = "report_example_microarray_binary.pdf",
           info.sample = as.data.frame(colData(ALCL)[, 8:10]),
           cex.names = 0.3, print.annot = TRUE)
