rm(list = ls())
gc()
library(CGGMR)

# Generate some covariance data
set.seed(1)
data = generateCovariance(7, 6)

# Compute weight matrix
W1 = cggmWeights(data$sample, phi = 1, k = 2)

lambdas_1 = seq(0, 0.25, 0.05)
lambdas_2 = c(0.18, 0.26)

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
