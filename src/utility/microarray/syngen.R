require(madsim)

working_dir <- file.path("R:/GeneAnalysis/data")

# NOTE: Please use the follwing settings
# 1. For increasing variability of normal expressed genes 
# use the following setting: lambda2 <- 2
# 2. For increasing variability of weekly expressed genes 
# use the following setting: lambda2 <- 0.1

## Step 1: Load a sample of real microarray data
data(madsim_test)

## Step 2: Set parameters settings
# Number of genes to sample
n <- 1100
# Number of control samples 
m1 <- 20
# Number of test samples 
m2 <- 20
# Beta distribution shape parameters
shape1 <- 2; shape2 <- 4
# Lower bound of log2 intensity values (default 4)
lb <- 2
# Upper bound of log2 intensity values (default 14)
ub <- 16
# This parameter controls the number of differentially 
# expressed genes the user would like to have in the 
# dataset (default 0.02)
pde <- 0.1
# This parameter results in nearly the same 
# number of up/down-regulated probes when sym = 0.5
sym <- 0.5
# Rate parameter for the exponential distribution
# Increasing lambda1 will lead to more variability 
# for weakly expressed genes and a small variability 
# for strongly expressed genes
# TODO: default 0.13
lambda1 <- 0.13
# Rate parameter for lambda * exp^(-lambda)
# A higher lambda2 value will lead to a small number 
# of genes having a shift greater than mean
# TODO: default 2
# lambda2 <- 0.5
# Mean parameter for the normal distribution
muminde <- 1
# Std parameter for the normal distribution
sdde <- 0.5
# This parameter represents a normal distribution 
# standard deviation for additive noise
sdn <- 0.4
# Computer random number initialization
rseed <- 50

## Step 3: Iterate to create two types of simulated data synthetic data
for (val in list(c(2, "simulated_normal"), c(0.1, "simulated_weak")))
  # for (val in list(c(0.1, "simulated_weak")))
{
  lambda2 <- as.numeric(val[1])
  file_name <- val[2]

  # Create two dataframes
  fparams <- data.frame(m1 = m1, m2 = m2, shape1 = shape1, shape2 = shape2,
                        lb = lb, ub = ub, pde = pde, sym = sym)
  dparams <- data.frame(lambda1 = lambda1, lambda2 = lambda2, muminde = muminde,
                        sdde = sdde)

  ## Step 4: Use true affymetrix data to generate synthetic data
  synData <- madsim(mdata = madsim_test, n = n, ratio = 0, fparams = fparams,
                    dparams = dparams, sdn = sdn, rseed = rseed)
  df <- as.data.frame(t(synData[["xdata"]]))
  rownames(df) <- NULL
  feature_names <- colnames(df)
  df <- cbind(class = c(rep(0, m1), rep(1, m2)), df)

  boxplot(df[which(df$class == 1), !(names(df) %in% "class")])

  ## Step 5: Save dataframe
  write.table(df, file = file.path(working_dir, paste(file_name, ".csv", sep = "")),
              sep = ",", quote = FALSE, row.names = FALSE)
  df_features <- as.data.frame(synData[["xid"]])
  rownames(df_features) <- feature_names
  colnames(df_features) <- "regulated"
  write.table(df_features,
              file = file.path(working_dir, paste(file_name, "_features.csv", sep = "")),
              sep = ",", quote = FALSE, row.names = FALSE)
}
