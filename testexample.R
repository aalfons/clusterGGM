library(mvtnorm)
library(clusterglasso)

k <- 10
n <- 100
X <- rmvnorm(n, sigma = diag(1, k))
lambda1 <- 0
lambda2 <- 0
fit <- taglasso(X, A = cbind(diag(1, k), rep(1, k)), pendiag = F,  lambda1, lambda2)
fit$omega_full


fit2 <- clusterglasso(X, lambda1 = lambda1, lambda2 = lambda2)
round(fit2$omega_full, 3)
