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
Omega = solve(cov(X))

k = 100
res = matrix(0, nrow = nrow(Omega) * k, 4)

# Clusterglasso
lambda1 = 0.00
lambda2 = 0.01

# First fit
fit = clusterglasso(X, lambda1 = lambda1, lambda2 = lambda2,
                      knn_weights = TRUE, phi = 0.5, knn = 1,
                      refitting = FALSE, adaptive = TRUE, eps_fusions = 1e-4)
C0 = cmdscale(dist(fit$fit$c))
plot(C0, col = c(1:10))

# Get sequence of fits
for (i in 1:k) {
  fit = clusterglasso(X, lambda1 = lambda1, lambda2 = lambda2,
                      knn_weights = TRUE, phi = 0.5, knn = 1,
                      refitting = FALSE, adaptive = TRUE)
  C1 = cmdscale(dist(fit$fit$c))

  eps = 0.10
  print(abs((C0 - C1)) > eps)
  C1[abs((C0 - C1)) > eps] = C1[abs((C0 - C1)) > eps] * (-1)
  print(C1)

  res[((i - 1) * nrow(C1) + 1):(i * nrow(C1)), c(3, 4)] = C1
  res[((i - 1) * nrow(C1) + 1):(i * nrow(C1)), 2] = lambda2
  res[((i - 1) * nrow(C1) + 1):(i * nrow(C1)), 1] = c(1:nrow(C1))

  points(C1, col = c(1:10), pch=16)

  lambda2 = lambda2 + 0.001
  C0 = C1
}

write.table(res, file = "res_test.csv", sep = ",", row.names = F, col.names = F)
