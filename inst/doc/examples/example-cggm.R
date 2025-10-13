## CGGM can be used to estimate precision matrices
# Generate the data
set.seed(3)
Theta <- matrix(
  c(2, 1, 0, 0,
    1, 2, 0, 0,
    0, 0, 4, 1,
    0, 0, 1, 4),
  nrow = 4
)
X <- mvtnorm::rmvnorm(n = 100, sigma = solve(Theta))
S <- cov(X)

# Compute the weight matrix for the clusterpath (clustering) weights
W_cpath <- clusterpath_weights(S, phi = 1, k = 2)

# Compute the weight matrix for the lasso (sparsity) weights
W_lasso <- lasso_weights(S)

# Set lambdas for the clusterpath penalty
lambdas <- seq(0, 0.2, 0.01)

# Estimate the precision matrix for each value for lambda and a fixed value for
# lambda_lasso, the regularization parameter for sparsity
res <- cggm(
  S, W_cpath = W_cpath, lambda_cpath = lambdas, W_lasso = W_lasso,
  lambda_lasso = 0.2
)

# The index of the first value for lambda for which there are 2 clusters
res$cluster_solution_index[2]

# Accessor function that retrieve the solution with 2 clusters
get_Theta(res, index = res$cluster_solution_index[2])
get_clusters(res, index = res$cluster_solution_index[2])

# Often, it is not clear which values for lambda make up the right sequence.
# The sequence can be expanded automatically
res <- cggm(
  S, W_cpath = W_cpath, lambda_cpath = lambdas, W_lasso = W_lasso,
  lambda_lasso = 0.2, expand = TRUE
)

# A solution with 2 clusters
get_Theta(res, index = res$cluster_solution_index[2])
get_clusters(res, index = res$cluster_solution_index[2])

## CGGM can also be used to estimate covariance matrices
# Generate the data
set.seed(3)
Sigma <- matrix(
  c(2, 1, 0, 0,
    1, 2, 0, 0,
    0, 0, 4, 1,
    0, 0, 1, 4),
  nrow = 4
)
X <- mvtnorm::rmvnorm(n = 100, sigma = Sigma)
S <- cov(X)

# Compute the weight matrix for the clusterpath (clustering) weights. The
# input is now the sample precision matrix
W_cpath <- clusterpath_weights(solve(S), phi = 1, k = 2)

# Compute the weight matrix for the lasso (sparsity) weights
W_lasso <- lasso_weights(solve(S))

# Set lambdas for the clusterpath penalty
lambdas <- seq(0, 0.2, 0.01)

# Use the sample precision matrix to estimate the covariance matrix for each
# value for lambda and a fixed value for lambda_lasso, the regularization
# parameter for sparsity
res <- cggm(
  solve(S), W_cpath = W_cpath, lambda_cpath = lambdas, W_lasso = W_lasso,
  lambda_lasso = 0.2, expand = TRUE
)

# A solution with 2 clusters
get_Theta(res, index = res$cluster_solution_index[2])
get_clusters(res, index = res$cluster_solution_index[2])