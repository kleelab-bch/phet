# R code for most statistics

calmoststatistics <- function(sample, num_control, num_case) {
  result <- array(0, dim = c(dim(sample)[1], num_case))
  temp <- generateorder(num_case)
  mu <- temp[, 1]
  std <- temp[, 2]
  n <- num_control + num_case
  sample_control <- sample[, 1:num_control]
  sample_case <- sample[, (num_control + 1):n]
  medx <- apply(sample_control, 1, median)
  medy <- apply(sample_case, 1, median)
  diffxmedx <- abs(sample_control - medx)
  diffymedy <- abs(sample_case - medy)
  stdforeachrow <- (apply(cbind(diffxmedx, diffymedy), 1, median) + 0.01) * 1.4826
  for (ktest in 1:num_case)
  {
    if (ktest > 1) { meanoutlier <- apply(sample_case[, 1:ktest], 1, mean) }
    else { meanoutlier <- sample_case[, 1] }
    result[, ktest] <- ((meanoutlier - medx) * ktest / stdforeachrow - mu[ktest]) / std[ktest]
  }
  mystatistics <- apply(result, 1, max)
}

# This file contains two functions, generateorder use simulation to generate 
# the mean and standard deviation used in standardizing M_{ik} for differentk.
#The other function generateonesim is used to generate simulated gene 
#expression data, 

generateorder <- function(n) {
  sample <- rnorm(1000 * n)
  msample <- matrix(sample, nrow = 1000)
  ordered <- apply(msample, MARGIN = 1, FUN = sort, decreasing = T)
  cumsumordered <- apply(ordered, 2, cumsum)
  a <- apply(cumsumordered, 1, mean)
  cumsumordered <- cumsumordered - a
  b <- apply(cumsumordered, 1, sd)
  cbind(a, b)
}

#simulate one set with num_features genes and n=num_control+num_case arrays, mu is the magnitude of 
#difference for two samples, k is number of outliers, each row in the disease
#sample is finally sorted, h1 is number of rows that are DE
generateexpression <- function(num_control, num_case, num_features, mu, k, h1 = 1) {
  n <- num_control + num_case # number of arrays
  samplex <- matrix(rnorm(num_features * num_control), nrow = num_features)
  sampley <- matrix(rnorm(num_features * num_case), nrow = num_features)
  if (h1 >= 1)
  { sampley[1:h1, 1:k] = sampley[1:h1, 1:k] + mu }
  sampley <- t(apply(sampley, 1, sort, decreasing = T))
  cbind(samplex, sampley)
}

# Change the following line to simulate under different parameters
num_control <- 20
num_case <- 20
num_features <- 100
mu <- 2
k <- 10
# Generate data from null distribution
sample0 <- generateexpression(num_control, num_case, num_features, mu, k, 0)
# Generate data from alternative distribution
sample1 <- generateexpression(num_control, num_case, num_features, mu, k, num_features)

#most statistics
mystat0 <- calmoststatistics(sample0, num_control, num_case)
mystat1 <- calmoststatistics(sample1, num_control, num_case)



