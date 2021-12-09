library(mvtnorm)
library(clusterglasso)


# Generate a precision matrix with a clear block structure
sigma_inv = matrix(0, nrow = 10, ncol = 10)
for (i in 1:10) {
  for (j in 1:10) {
    if ((i <= 5 && j <= 5) || (i > 5 && j > 5)) {
      sigma_inv[i, j] = 1
    }
  }
}
sigma_inv = (sigma_inv + diag(10))

# Invert it to obtain the covariance matrix used to generate the data
sigma = solve(sigma_inv)
set.seed(42)
X = rmvnorm(100, sigma = sigma)

# Alternative data
X = matrix(0, nrow = 1000000, ncol = 4)
X[, 1] = rnorm(nrow(X), mean = 0, sd = 1)
X[, 2] = X[, 1] + rnorm(nrow(X), mean = 0, sd = 1)
X[, 3] = X[, 2] + rnorm(nrow(X), mean = 0, sd = 1)
X[, 4] = X[, 2] + rnorm(nrow(X), mean = 0, sd = 1)
Omega = solve(cov(X))


# Clusterglasso
lambda1 = 0.05
lambda2 = 0.05

fit2 <- clusterglasso(X, lambda1 = lambda1, lambda2 = lambda2,
                      knn_weights = TRUE, phi = 0.5, knn = 1,
                      refitting = FALSE, adaptive = TRUE)
round(fit2$omega_full, 2)
fit2$cluster

# Print the adjacency matrix derived from the weight matrix, each nonzero weight
# is represented by a 1
print((fit2$W_aggregation > 0) * 1)

chol(Omega)
