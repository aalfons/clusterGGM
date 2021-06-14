rm(list=ls())
#### Rscript : How to make use of the clusterglasso functions in the scripts for simulation study ####

#### Libraries ####
library(mvtnorm)
library(clusterglasso)
library(BayesianOptimization)

#### Generate the data ####
n = 100
p = 10
precision = matrix(0, nrow = p, ncol = p)
for (i in 1:p) {
  for (j in 1:p) {
    if ((i <= p/2 && j <= p/2) || (i > p/2 && j > p/2)) {
      precision[i, j] = 1
    }
  }
}
precision = (precision + diag(p))
sigma = solve(precision)
set.seed(42)
X = rmvnorm(n, sigma = sigma)



#### clusterglasso ####
select_tuning <- tuning_clusterglasso(X = X, seed_cv = 42)
fit <- clusterglasso(X = X, lambda1 = select_tuning$lambda1_opt , lambda2 = select_tuning$lambda2_opt) #IW: I've included the computation of the weight matrix inside the function clusterglasso as current default
#### How to evaluate performance? ####
# Estimation accuracy: use output slot fit$omega_full (estimate of the pxp precision matrix)
# Aggregation performance: use output slot fit$cluster (p-dimensional vector indictating cluster membership)
# Sparsity recognition : use output slot fit$sparsity (0/1 matrix : 1 for non-zeros)
