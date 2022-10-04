require(UpSetR)

working_dir <- file.path("R:/GeneAnalysis/result")
df <- read.table(file.path(working_dir, "upset.csv"),
                 header = TRUE, sep = ",", check.names = FALSE,
                 stringsAsFactors = FALSE)
classes <- df[, 1]
drop_cols <- c("")
featureNames <- colnames(df)[!(names(df) %in% drop_cols)]
df <- as.data.frame(t(df[, featureNames]))
colnames(df) <- classes

upset(df, nsets = 10, nintersects = 30, mb.ratio = c(0.5, 0.5),
      order.by = c("freq", "degree"), decreasing = c(TRUE, FALSE))