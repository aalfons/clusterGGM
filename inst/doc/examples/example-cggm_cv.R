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

# The best solution has 2 clusters
get_Theta(res_cv)
get_clusters(res_cv)

# The best result was obtained using the refitting step
res_cv$best