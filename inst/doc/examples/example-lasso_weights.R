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

# Compute the weight matrix for the lasso (sparsity) weights
W_lasso <- lasso_weights(S)