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

# Estimate the precision matrix while automatically expanding the sequence of
# values for lambda
res <- cggm(
  S, W_cpath = W_cpath, lambda_cpath = lambdas, W_lasso = W_lasso,
  lambda_lasso = 0.2, expand = TRUE
)

# Precision matrix with 2 clusters
get_Theta(res, index = res$cluster_solution_index[2])

# Apply the refitting step to the results, estimating the precision matrix
# based on the clustering and sparsity patterns but without additional
# shrinkage
res_refit <- cggm_refit(res)

# Precision matrix with 2 clusters
get_Theta(res_refit, index = res_refit$cluster_solution_index[2])

# Use cross-validation to select the tuning parameters
res_cv <- cggm_cv(
    X = X,
    tune_grid = expand.grid(
        phi = c(0.5, 1.0),
        k = c(2),
        lambda_lasso = c(0, 0.02, 0.05)
    ),
    folds = cv_folds(nrow(X), 5)
)

# The best solution for the precision matrix
get_Theta(res_cv)