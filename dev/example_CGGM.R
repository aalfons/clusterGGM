# ************************************
# Author: Andreas Alfons
#         Erasmus University Rotterdam
# ************************************


## load packages and functions
library("clusterGGM")
library("mvtnorm")
source("dev/evaluation_criteria.R")
source("dev/utils.R")


## control parameters for data generation
seed <- 20231116  # seed for the random number generator
n <- 120L         # number of observations
p <- 15L          # number of variables
B <- 3L           # number of blocks
theta_w <- 0.5    # (off-diagonal) elements within blocks
theta_b <- 0.25   # non-zero elements between blocks

## additional parameters
blocks <- sort(rep(1:B, length.out = p))  # works if p is not a multiple of B
p_b <- tabulate(blocks)                   # number of variables per block

## grid for tuning parameters
# sparsity parameter will be set based on smallest lambda that sets all
# elements to 0 in graphical lasso
lambda_frac <- lambda_grid(c(0, 1), n_points = 10, factor = 2)
# additional tuning parameters for aggregation penalty in CGGM
phi <- 1            # tuning parameter for weights
k <- c(1L, 3L, 5L)  # number of nearest neighbors for weights

## other control parameters for cross-validation
K <- 3  # number of folds

## control parameters for evaluation
tol <- 1e-05


## utility functions for data generation

# function to generate diagonal blocks of precision matrix
generate_diagonal_block <- function(d, theta) {
  mat <- matrix(theta, nrow = d, ncol = d)
  diag(mat) <- 1
  mat
}

# function to generate offdiagonal blocks of precision matrix
generate_offdiagonal_block <- function(nrow, ncol, theta) {
  matrix(theta, nrow = nrow, ncol = ncol)
}


## generate data

# set seed of random number generator for reproducibility
set.seed(seed)

# initialize precision matrix as a list of lists:
# (outer list corresponds to columns, inner list to rows of the given column)
Theta <- replicate(B, replicate(B, NULL, simplify = FALSE),
                   simplify = FALSE)
# loop over indices of blocks in the final precision matrix, and generate
# those blocks following the R convention of building a matrix by column
# (i is the index of the row, j is the index of the column)
for (j in seq_len(B)) {
  for (i in seq_len(B)) {
    if (i == j) Theta[[j]][[i]] <- generate_diagonal_block(p_b[i], theta_w)
    else if (i == j+1) {
      Theta[[j]][[i]] <- generate_offdiagonal_block(p_b[i], p_b[j], theta_b)
    } else if (i > j+1) {
      Theta[[j]][[i]] <- generate_offdiagonal_block(p_b[i], p_b[j], 0)
    } else Theta[[j]][[i]] <- t(Theta[[i]][[j]])  # upper diagonal blocks
  }
}
# put precision matrix together
Theta <- do.call(cbind, lapply(Theta, function(column) do.call(rbind, column)))

## generate covariance matrix by inverting
Sigma <- solve(Theta)

## generate training data
X <- rmvnorm(n, sigma = Sigma)

## compute sample covariance matrix
S <- cov(X)

## estimate the smallest lambda1 that sets everything to 0 in glasso
S_offdiagonal <- S - diag(p) * diag(S)
lambda_max_glasso <- max(max(S_offdiagonal), -min(S_offdiagonal))

## generate folds for cross-validation
folds <- cv_folds(n, K = K)


## apply CGGM with and without refitting

# construct grid of tuning parameter values
# (aggregation parameter is determined automatically for each combination)
tune_grid <- expand.grid(phi = phi, k = k,
                         lambda_lasso = lambda_frac * lambda_max_glasso)
# perform grid search to find tuning parameters and re-fit with optimal
# tuning parameters
fit_cv <- cggm_cv(X, tune_grid = tune_grid, folds = folds)
# extract estimates with optimal parameters
Theta_raw <- get_Theta(fit_cv$fit$final,
                       index = fit_cv$fit$opt_index)
Theta_refit <- get_Theta(fit_cv$refit$final,
                         index = fit_cv$refit$opt_index)
Theta_best <- get_Theta(fit_cv)
# extract clusters with optimal parameters
clusters_raw <- get_clusters(fit_cv$fit$final,
                             index = fit_cv$fit$opt_index)
clusters_refit <- get_clusters(fit_cv$refit$final,
                               index = fit_cv$refit$opt_index)
clusters_best <- get_clusters(fit_cv)
# compute evaluation criteria
error <- c(Frobenius(Theta, Theta_raw),
           Frobenius(Theta, Theta_refit),
           Frobenius(Theta, Theta_best))
NLL <- c(neg_log_lik(Sigma, Theta_raw),
         neg_log_lik(Sigma, Theta_refit),
         neg_log_lik(Sigma, Theta_best))
B_hat <- c(length(unique(clusters_raw)),
           length(unique(clusters_refit)),
           length(unique(clusters_best)))
ARI <- c(adjusted_Rand_index(blocks, clusters_raw),
         adjusted_Rand_index(blocks, clusters_refit),
         adjusted_Rand_index(blocks, clusters_best))
FP_rate <- c(FPR(Theta, Theta_raw, tol = tol),
             FPR(Theta, Theta_refit, tol = tol),
             FPR(Theta, Theta_best, tol = tol))
FN_rate <- c(FNR(Theta, Theta_raw, tol = tol),
             FNR(Theta, Theta_refit, tol = tol),
             FNR(Theta, Theta_best, tol = tol))
# extract optimal tuning parameters
lambda_opt <- c(fit_cv$fit$opt_tune$lambda,
                fit_cv$refit$opt_tune$lambda)
lambda_lasso_opt <- c(fit_cv$fit$opt_tune$lambda_lasso,
                      fit_cv$refit$opt_tune$lambda_lasso)
phi_opt <- c(fit_cv$fit$opt_tune$phi,
             fit_cv$refit$opt_tune$phi)
k_opt <- c(fit_cv$fit$opt_tune$k,
           fit_cv$refit$opt_tune$k)
if (fit_cv$best == "fit") {
  lambda_opt <- lambda_opt[c(1, 2, 1)]
  lambda_lasso_opt <- lambda_lasso_opt[c(1, 2, 1)]
  phi_opt <- phi_opt[c(1, 2, 1)]
  k_opt <- k_opt[c(1, 2, 1)]
} else {
  lambda_opt <- lambda_opt[c(1, 2, 2)]
  lambda_lasso_opt <- lambda_lasso_opt[c(1, 2, 2)]
  phi_opt <- phi_opt[c(1, 2, 2)]
  k_opt <- k_opt[c(1, 2, 2)]
}
# compute evaluation criteria and combine into data frame
labels <- paste("CGGM", c("raw", "refit", "best"), sep = "-")
data.frame(n = n, p = p, B = B, Method = labels,
           lambda_aggregation = lambda_opt,
           lambda_sparsity = lambda_lasso_opt,
           phi = phi_opt, k = k_opt, Error = error,
           NLL = NLL, B_hat = B_hat, ARI = ARI,
           FPR = FP_rate, FNR = FN_rate,
           stringsAsFactors = FALSE)
