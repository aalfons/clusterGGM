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

## other control parameters for cross-validation
K <- 3  # number of folds

## control parameters for evaluation
tol <- 1e-05  # tag-lasso cannot detect zeros at tolerance 1e-06


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


## apply tag-lasso with realistic tree hierarchy

# construct tree aggregation matrix
A <- get_tree_matrices(blocks)[[2]]

# find aggregation parameter that yields maximal aggregation
lambda1_max_TAGL <- binary_search(target = "lambda1", fun = taglasso,
                                  X = X, A = A, lambda2 = 0, refitting = TRUE,
                                  adaptive = FALSE)
# perform grid search to find regularization parameters
opt_lambda <- cv_taglasso(X, A, lambda1 = lambda_frac * lambda1_max_TAGL,
                          lambda2 = lambda_frac * lambda_max_glasso,
                          folds = folds, type = "grid",
                          refitting = TRUE, adaptive = FALSE)
# re-fit with optimal regularization parameters
fit_opt <- taglasso(X, A, lambda1 = opt_lambda$lambda1_opt,
                    lambda2 = opt_lambda$lambda2_opt,
                    refitting = TRUE, adaptive = FALSE)
# compute evaluation criteria and combine into data frame
data.frame(n = n, p = p, B = B, Method = "tag-lasso",
           lambda_aggregation = opt_lambda$lambda1_opt,
           lambda_sparsity = opt_lambda$lambda2_opt,
           Error = Frobenius(Theta, fit_opt$omega_full),
           NLL = neg_log_lik(Sigma, fit_opt$omega_full),
           B_hat = length(unique(fit_opt$cluster)),
           ARI = adjusted_Rand_index(blocks, fit_opt$cluster),
           FPR = FPR(Theta, fit_opt$omega_full, tol = tol),
           FNR = FNR(Theta, fit_opt$omega_full, tol = tol),
           stringsAsFactors = FALSE)
