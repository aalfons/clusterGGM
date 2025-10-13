## CGGM can be used to estimate a clustered precision matrix

# Generate data
set.seed(3)
Theta <- matrix(
  c(2, 1, 0, 0,
    1, 2, 0, 0,
    0, 0, 4, 1,
    0, 0, 1, 4),
  nrow = 4
)
X <- mvtnorm::rmvnorm(n = 100, sigma = solve(Theta))

# Estimate the covariance matrix
S <- cov(X)

# Compute the weight matrix for the clusterpath (clustering) weights
W_cpath <- clusterpath_weights(S, phi = 1, k = 2)

# Compute the weight matrix for the lasso (sparsity) weights
W_lasso <- lasso_weights(S)

# Set lambdas for the clusterpath penalty
lambdas <- seq(0, 0.2, 0.01)

# Estimate the precision matrix for each value for lambda and a fixed
# value for lambda_lasso, the regularization parameter for sparsity
fit <- cggm(S, W_cpath = W_cpath, lambda_cpath = lambdas,
            W_lasso = W_lasso, lambda_lasso = 0.2)

# The index of the first value for lambda for which there are 2 clusters
keep <- fit$cluster_solution_index[2]

# Accessor function that retrieve the solution with 2 clusters
get_Theta(fit, index = keep)
get_clusters(fit, index = keep)


# Often, it is not clear which values for lambda make up the
# right sequence. The sequence can be expanded automatically.
fit <- cggm(
  S, W_cpath = W_cpath, lambda_cpath = lambdas, W_lasso = W_lasso,
  lambda_lasso = 0.2, expand = TRUE
)

# A solution with 2 clusters
keep <- fit$cluster_solution_index[2]
get_Theta(fit, index = keep)
get_clusters(fit, index = keep)


## CGGM can also be used to estimate a clustered covariance matrix
# Generate data
set.seed(3)
Sigma <- matrix(
  c(2, 1, 0, 0,
    1, 2, 0, 0,
    0, 0, 4, 1,
    0, 0, 1, 4),
  nrow = 4
)
X <- mvtnorm::rmvnorm(n = 100, sigma = Sigma)

# Estimate the covariance matrix and compute its inverse
S <- cov(X)
S_inv <- solve(S)

# Compute the weight matrix for the clusterpath (clustering) weights.
# The input is now the sample precision matrix.
W_cpath <- clusterpath_weights(S_inv, phi = 1, k = 2)

# Compute the weight matrix for the lasso (sparsity) weights.
# The input is again the sample precision matrix.
W_lasso <- lasso_weights(S_inv)

# Set lambdas for the clusterpath penalty
lambdas <- seq(0, 0.2, 0.01)

# Use the sample precision matrix to estimate the covariance matrix
# for each value for lambda and a fixed value for lambda_lasso, the
# regularization parameter for sparsity
fit <- cggm(S_inv, W_cpath = W_cpath, lambda_cpath = lambdas,
            W_lasso = W_lasso, lambda_lasso = 0.2, expand = TRUE)

# A solution with 2 clusters
keep <- fit$cluster_solution_index[2]
get_Theta(fit, index = keep)
get_clusters(fit, index = keep)
