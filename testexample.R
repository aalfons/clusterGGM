library(mvtnorm)
library(clusterglasso)

k <- 10
n <- 100
X <- rmvnorm(n, sigma = diag(1, k))
lambda1 <- 0
lambda2 <- 0
fit <- taglasso(X, A = cbind(diag(1, k), rep(1, k)), pendiag = F,  lambda1, lambda2)
fit$omega_full
