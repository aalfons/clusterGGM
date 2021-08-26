library(mvtnorm)
library(clusterglasso)

# Check to see if the code still works (and behaves as expected) with a weight
# matrix full of ones
k <- 10
n <- 100
X <- rmvnorm(n, sigma = diag(1, k))
W <- matrix(1, nrow = k, ncol = k)
lambda1 <- 0
lambda2 <- 1

fit <- taglasso(X, A = cbind(diag(1, k), rep(1, k)), pendiag = F,  lambda1, lambda2)
fit$omega_full

fit2 <- clusterglasso(X, W, lambda1 = lambda1, lambda2 = lambda2)
round(fit2$omega_full, 3)




################################################################################
# To test the efficacy of the new weights
################################################################################

# Function to compute distances based on what was discussed May 20th
distance <- function(omega) {
  k = ncol(omega)
  result = matrix(0, ncol = k, nrow = k)

  for (i in 1:k) {
    for (j in 1:k) {
      result[i, j] = sum((omega[i, -c(i, j)] - omega[j, -c(i, j)])^2)^0.5
    }
  }

  return(result)
}

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
X = rmvnorm(n, sigma = sigma)

# Compute a different weight matrix based on the distances
W = solve(cov(X))
W = distance(W)
W = exp(-0.5 * W^2)

# W looks very favorable, if this doesn't work well something must be wrong
round(W, 3)

# EXAMPLE 1: With the weights, the clustering of the precision matrix is
# retrieved without any issues based on the first 3 decimal places
lambda1 <- 0
lambda2 <- 0.14

t1 = Sys.time()
fit2 <- clusterglasso(X, W, lambda1 = lambda1, lambda2 = lambda2)
print(Sys.time() - t1)
round(fit2$omega_full, 3)
fit2$cluster

# EXAMPLE 2: However, it appears that cluster hierarchy is somewhat violated
# based on the rounding at 5 decimals we are using right now
lambda1 <- 0
lambda2 <- 0.15

fit2 <- clusterglasso(X, W, lambda1 = lambda1, lambda2 = lambda2)
round(fit2$omega_full, 3)
fit2$cluster

# EXAMPLE 3: The actual correct clustering based on the convergence of 5 decimal
# places is obtained with lambda2 = 0.53, much larger than the 0.14 where we
# already identified a clear block structure. So, numerical precision comes at
# the cost of a larger bias
lambda1 <- 0
lambda2 <- 0.53

fit2 <- clusterglasso(X, W, lambda1 = lambda1, lambda2 = lambda2)
round(fit2$omega_full, 3)
fit2$cluster

# EXAMPLE 4: Setting lambda1 to something larger than zero shows great results,
# which is of course expected with such an easy data generating process, but is
# definitely encouraging
lambda1 <- 0.01
lambda2 <- 0.14

fit2 <- clusterglasso(X, W, lambda1 = lambda1, lambda2 = lambda2)
round(fit2$omega_full, 3)
fit2$cluster

