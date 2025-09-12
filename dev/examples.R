# Clear environment to prevent mistakes
rm(list = ls())
gc()

# Load packages
library("clusterGGM")
library("igraph")


## Additional functions

# Generate R and A matrix
#' @importFrom stats rnorm
.generate_RA <- function(K)
{
  # Generate symmetric matrix R
  R = matrix(rnorm(K * K * 2), nrow = K)
  for (i in 1:K) {
    R[i, ] = R[i, ] - mean(R[i, ])
  }
  R = R %*% t(R) / (2 * K)

  # Generate vector A with positive elements using a truncated normal
  # distribution
  A = rnorm(K, mean = 1, sd = 0.5)
  for (i in 1:K) {
    while (A[i] <= 0) {
      A[i] = rnorm(1, mean = 1, sd = 0.5)
    }
  }

  return(list("R" = R, "A" = A))
}

#' Generate Covariance Matrix
#'
#' This function generates a \code{n_vars} by \code{n_vars} covariance matrix in
#' which the rows/columns are genereated from \code{n_clusters} clusters.
#'
#' @param n_vars The number of variables in the covariance matrix.
#' @param n_clusters The number of clusters in the covariance matrix.
#' @param n_draws The number of observations that should be drawn from a
#' multivariate normal distribution with the true covariance matrix.
#' @param shuffle Logical indicating whether to shuffle the covariance matrix.
#' Defaults to \code{FALSE}.
#'
#' @return A list containing the true covariance matrix, the sample covariance
#' matrix, the true cluster labels, and the sample data set generated using the
#' true covariance matrix.
#'
#' @importFrom mvtnorm rmvnorm
#' @export
generate_covariance <- function(n_vars, n_clusters, n_draws = 100 * n_vars,
                                shuffle = FALSE)
{
  if (n_clusters > n_vars) {
    stop(paste("The number of clusters must be smaller than or equal to",
               "the number of variables (test failed: c_clusters <=",
               " n_vars)."))
  }

  if (shuffle) {
    warning("Shuffle is not implemented yet.")
  }

  # Membership vector
  u = c(1:n_clusters)
  u = c(u, sample(u, n_vars - n_clusters, replace = TRUE))
  u = sort(u)

  # Generate clustered data
  Ra = .generate_RA(n_clusters)
  R = Ra$R
  A = Ra$A

  # Compute Theta
  Theta = R[u, u] + diag(A[u])

  # True covariance matrix
  Sigma = solve(Theta)

  # Draw data
  data = mvtnorm::rmvnorm(n_draws, sigma = Sigma)

  # Compute sample covariance matrix
  S = cov(data)

  # Fill result
  result = list()
  result$sample = S
  result$true = Sigma
  result$clusters = u
  result$data = data

  return(result)
}


## Example 1
# Generate covariance matrix with a particular number of variables that are
# driven by an underlying cluster structure
set.seed(1)
data = generate_covariance(n_vars = 5, n_clusters = 2)

# The variable data contains a true covariance matrix, which we store in Sigma,
# a sample covariance matrix S, and the true cluster labels
Sigma = data$true
S = data$sample

# View true Theta and true cluster labels
print(solve(Sigma))
print(data$clusters)

# Compute weight matrix, based on exp(-phi * d(Theta_i, Theta_j)), with sparsity
# based on the 2 nearest neighbors: k = 2 (the dense matrix is only made sparse
# for k in a "sensible" range: k in [1, nrow(S) - 1]). By default a connected
# weight matrix is guaranteed by the argument connected = TRUE
W = clusterpath_weights(S, phi = 1, k = 2)

# Plot the weight matrix as a weighted graph
G = graph_from_adjacency_matrix(W, mode = "undirected", weighted = TRUE)
plot(G, edge.label = round(E(G)$weight, 3), layout = layout.circle(G))

# Set lambda
lambdas = seq(0, 0.1, 0.01)

# Testing the algorithm, ?cggm provides some explanation of the inputs
res = cggm(S, W_cpath = W, lambda_cpath = lambdas)

# The result contains a lot of information. Here we take a look at the number of
# clusters found after minimizing each instance of the loss function
res$cluster_counts

# We also have accessor functions for both Theta and the cluster labels. Here
# the solution for the last value of lambda is retrieved
get_Theta(res, index = res$n)
get_clusters(res, index = res$n)


## Example 2
# Often, it is not clear which values of lambda make up a "sensible" sequence
# with appropriate step sizes from one to the next. That is why cggm() also has
# and expand argument, which automatically finds the smallest number of clusters
# and adds values of lambda so that the difference between consecutive solutions
# is not too large. Where "too large" is determined by the max_difference
# argument
res = cggm(S, W, lambda_cpath = lambdas, expand = TRUE)

# Finally, we can refit the result without penalty but with cluster constraints,
# this is not very relevant right now, but may be useful later if we want to
# compare the solution for Theta to the true value
refit_res = cggm_refit(res)

# View the results
print(get_Theta(res, 16))               # Fitted
print(get_Theta(refit_res, 3))          # Refitted
print(solve(Sigma))                     # True

# We can see that there is only a small bias when comparing the fitted and
# refitted versions of Theta


## Example 3
# Perform k-fold CV to select the optimal value of phi, k, and lambda. This
# function is able to automatically tune the value of lambda if there is no
# column called lambda in the tune_grid data frame
folds = cv_folds(nrow(data$data), 5)
res_cv = cggm_cv(
    X = data$data,
    tune_grid = expand.grid(
        phi = c(0.5, 1.5), k = c(1, 2, 3), lambda = seq(0, 0.25, 0.01)
    ),
    folds = folds,
    verbose = 1
)

# See whether a fitted or refitted result was best
print(res_cv$best)

# The cluster labels after cross validation
print(get_clusters(res_cv))

# Theta after cross validation
print(get_Theta(res_cv))

# It is also possible to specify which result should be returned: "fit" or
# "refit". This is done via the which argument
print(get_clusters(res_cv, which = "fit"))
print(get_Theta(res_cv, which = "fit"))


## Example 4
# Perform k-fold CV with automatic lambda tuning. This does take longer than
# providing a grid for lambda yourself
res_cv = cggm_cv(
    X = data$data,
    tune_grid = expand.grid(
        phi = c(0.5, 1.5), k = c(1, 2, 3)
    ),
    folds = folds,
    verbose = 1
)

# See whether a fitted or refitted result was best
print(res_cv$best)

# The cluster labels after cross validation
print(get_clusters(res_cv))

# Theta after cross validation
print(get_Theta(res_cv))

## Example 6
# Next, an example of a type of problem that distinguishes CGGM from other
# methods: one where the cluster structure is only apparent from the values on
# the diagonal of Theta.
Theta = matrix(c(2, 1, 1, 1,
                 1, 2, 1, 1,
                 1, 1, 4, 1,
                 1, 1, 1, 4),
               nrow = 4)
X = mvtnorm::rmvnorm(n = 200, sigma = solve(Theta))

# Apply cross validation
folds = cv_folds(nrow(X), 5)
res_cv = cggm_cv(
    X = X,
    tune_grid = expand.grid(
        phi = c(0.5, 1.25, 2.0), k = c(1, 2, 3)
    ),
    folds = folds,
    verbose = 1
)

# See whether a fitted or refitted result was best
print(res_cv$best)

# The cluster labels after cross validation
print(get_clusters(res_cv))

# Theta after cross validation
print(get_Theta(res_cv))


## Example 7
# It is also possible to use the sample precision matrix as input to the
# optimization algorithm. For example, here we compute the weight matrix,
# compute a clusterpath, and refit the result while using solve(S) as input to
# estimate the covariance matrix instead of the precision matrix.
Theta_sample = solve(cov(X))

# Compute weight matrix
W = clusterpath_weights(Theta_sample, phi = 1, k = 2)

# Compute clusterpath
res_Sigma = cggm(
    Theta_sample, W, lambda_cpath = seq(0, 0.5, 0.05), expand = TRUE
)

# Refit
refit_res_Sigma = cggm_refit(res_Sigma)

# The accessors still work
get_clusters(refit_res_Sigma, index = refit_res_Sigma$n - 1)
get_Theta(refit_res_Sigma, index = refit_res_Sigma$n - 1)

# The true covariance matrix
solve(Theta)


## Example 8
# Of course, we can also apply cross validation to get a similar result
res_Sigma_cv = cggm_cv(
    X = X,
    tune_grid = expand.grid(phi = c(0.5, 1.25, 2.0), k = c(1, 2, 3)),
    folds = folds,
    verbose = 1,
    estimate_Sigma = TRUE
)

# The cluster labels after cross validation
print(get_clusters(res_Sigma_cv))

# Theta after cross validation
print(get_Theta(res_Sigma_cv))

# The true covariance matrix
solve(Theta)


## Example 9
# This example concerns sparsity, we can also use cross validation to tune the
# lasso penalty parameter.
Theta = matrix(c(2, 1, 0, 0,
                 1, 2, 0, 0,
                 0, 0, 4, 1,
                 0, 0, 1, 4),
               nrow = 4)
set.seed(3)
X = mvtnorm::rmvnorm(n = 300, sigma = solve(Theta))

# Apply cross validation
folds = cv_folds(nrow(X), 5)
res_cv = cggm_cv(
    X = X,
    tune_grid = expand.grid(
        phi = c(0.5), k = c(1), lambda_lasso = c(0, 0.02, 0.05)
    ),
    folds = folds,
    verbose = 1
)

# See whether a fitted or refitted result was best
print(res_cv$best)

# The cluster labels after cross validation
print(get_clusters(res_cv))

# Theta after cross validation
print(get_Theta(res_cv))

# True Theta
print(Theta)
