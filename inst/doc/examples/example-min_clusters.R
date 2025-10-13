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
# without enforcing connectedness
W_cpath <- clusterpath_weights(S, phi = 1, k = 1, connected = FALSE)

# The smallest number of clusters is 2
min_clusters(W_cpath)


# Compute the weight matrix for the clusterpath (clustering) weights
# with enforcing connectedness (default behavior)
W_cpath <- clusterpath_weights(S, phi = 1, k = 1, connected = TRUE)

# The smallest number of clusters is 1
min_clusters(W_cpath)
