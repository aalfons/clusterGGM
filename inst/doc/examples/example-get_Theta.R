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

# Set values to be used for the aggregation parameter
lambdas <- seq(0, 0.2, by = 0.01)

# Estimate the precision matrix while automatically expanding
# the sequence of values for the aggregation parameter
fit <- cggm(S, W_cpath = W_cpath, lambda_cpath = lambdas,
            W_lasso = W_lasso, lambda_lasso = 0.2,
            expand = TRUE)

# Precision matrix with 2 clusters
get_Theta(fit, index = fit$cluster_solution_index[2])


# Apply the refitting step to the results, estimating the
# precision matrix based on the clustering and sparsity
# patterns but without additional shrinkage
refit <- cggm_refit(fit)

# Precision matrix with 2 clusters
get_Theta(refit, index = refit$cluster_solution_index[2])
