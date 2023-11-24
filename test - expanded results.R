rm(list = ls())
gc()
library(CGGMR)

# Generate some covariance data
set.seed(1)
data = generateCovariance(20, 15)

# Compute weight matrix
W1 = cggmWeights(data$sample, phi = 1, k = 2)

lambdas_1 = seq(0, 0.25, 0.05)
lambdas_2 = c(0.18, 0.26, 0.35)

# Compute solutions for the initial set of lambdas
result_1 = cggmNew(S = data$sample, W = W1, lambdas = lambdas_1, gss_tol = 1e-4,
                   conv_tol = 1e-7, fusion_threshold = NULL, max_iter = 100,
                   store_all_res = TRUE, verbose = 0)

result_2 = cggm_expand(result_1, lambdas_2)

result_3 = cggmNew(S = data$sample, W = W1, lambdas = result_2$lambdas,
                   gss_tol = 1e-4, conv_tol = 1e-7, fusion_threshold = NULL,
                   max_iter = 100, store_all_res = TRUE, verbose = 0)

plot(result_2$lambdas, result_2$losses, type = "l")
lines(result_3$lambdas, result_3$losses, type = "l", col = "red")

result_2$cluster_counts
result_3$cluster_counts

################################################################################
#
################################################################################

rm(list = ls())
gc()
library(CGGMR)

# Generate some covariance data
set.seed(1)
data = generateCovariance(20, 20)

# Compute weight matrix
W = cggmWeights(data$sample, phi = 1, k = 2)
min_clusters(W)

lambdas = seq(0, 1, 0.01)

# Compute solutions for the initial set of lambdas
result = cggmNew(S = data$sample, W = W, lambdas = lambdas,
                 store_all_res = TRUE)
plot(lambdas, result$losses, type = "l")
result$cluster_counts

# Compute difference in norms
diffnorms = rep(0, result$n - 1)

for (i in 2:result$n) {
    diffnorms[i - 1] = sqrt(sum((result$Theta[[i - 1]] - result$Theta[[i]])^2))
}

# Plot versus lambdas
plot(result$lambdas[-1], diffnorms, type = "l")

