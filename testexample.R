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


# EXAMPLE 1: dense weights does well
lambda1 <- 0.00
lambda2 <- 0.06

fit2 <- clusterglasso(X, lambda1 = lambda1, lambda2 = lambda2,
                      knn_weights = FALSE, phi = 0.5, knn = 3,
                      refitting = FALSE)
round(fit2$omega_full, 2)
fit2$cluster

# Print the adjacency matrix derived from the weight matrix, each nonzero weight
# is represented by a 1
print((fit2$W_aggregation > 0) * 1)


# EXAMPLE 2: sparse weights with 3 nearest neighbors
lambda1 <- 0.00
lambda2 <- 4.00

fit2 <- clusterglasso(X, lambda1 = lambda1, lambda2 = lambda2,
                      knn_weights = TRUE, phi = 0.5, knn = 3,
                      refitting = FALSE)
round(fit2$omega_full, 2)
fit2$cluster

# Print the adjacency matrix derived from the weight matrix, each nonzero weight
# is represented by a 1
print((fit2$W_aggregation > 0) * 1)


# EXAMPLE 3: sparse weights with 3 nearest neighbors and symmetric circulant
lambda1 <- 0.00
lambda2 <- 4.00

fit2 <- clusterglasso(X, lambda1 = lambda1, lambda2 = lambda2,
                      knn_weights = TRUE, knn_connect = TRUE, phi = 0.5,
                      knn = 3, refitting = FALSE)
round(fit2$omega_full, 2)
fit2$cluster

# Print the adjacency matrix derived from the weight matrix, each nonzero weight
# is represented by a 1
print((fit2$W_aggregation > 0) * 1)


# EXAMPLE 4: sparse weights with 5 nearest neighbors and small lambda2
lambda1 <- 0.00
lambda2 <- 0.06

fit2 <- clusterglasso(X, lambda1 = lambda1, lambda2 = lambda2,
                      knn_weights = TRUE, phi = 0.5, knn = 5,
                      refitting = FALSE)
round(fit2$omega_full, 2)
fit2$cluster

