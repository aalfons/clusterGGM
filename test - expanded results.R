rm(list = ls())
gc()
library(CGGMR)

# Generate some covariance data
set.seed(1)
data = generate_covariance(20, 15)

# Compute weight matrix
W1 = cggm_weights(data$sample, phi = 1, k = 2)

lambdas_1 = seq(0, 0.25, 0.05)
lambdas_2 = c(0.18, 0.26, 0.35)

# Compute solutions for the initial set of lambdas
result_1 = cggm(S = data$sample, W = W1, lambdas = lambdas_1, gss_tol = 1e-4,
                conv_tol = 1e-7, fusion_threshold = NULL, max_iter = 100,
                store_all_res = TRUE, verbose = 0)

result_2 = cggm_expand(result_1, lambdas_2)

result_3 = cggm(S = data$sample, W = W1, lambdas = result_2$lambdas,
                gss_tol = 1e-4, conv_tol = 1e-7, fusion_threshold = NULL,
                max_iter = 100, store_all_res = TRUE, verbose = 0)

plot(result_2$lambdas, result_2$losses, type = "l")
lines(result_3$lambdas, result_3$losses, type = "l", col = "red")

result_2$cluster_counts
result_3$cluster_counts

################################################################################
# Norm of the difference between consecutive solutions
################################################################################

rm(list = ls())
gc()
library(CGGMR)

# Generate some covariance data
set.seed(1)
data = generate_covariance(20, 20)

# Compute weight matrix
W = cggm_weights(data$sample, phi = 1, k = 2)
min_clusters(W)

lambdas = seq(0, 1, 0.01)

# Compute solutions for the initial set of lambdas
result = cggm(S = data$sample, W = W, lambdas = lambdas, store_all_res = TRUE)
plot(lambdas, result$losses, type = "l")
result$cluster_counts

# Compute difference in norms
diffnorms = rep(0, result$n - 1)

for (i in 2:result$n) {
    diffnorms[i - 1] = norm(result$Theta[[i - 1]] - result$Theta[[i]], "F")
}

# Plot versus lambdas
plot(result$lambdas[-1], diffnorms, type = "l")


################################################################################
# Find minimum number of clusters
################################################################################

rm(list = ls())
gc()
library(CGGMR)

# Generate some covariance data
set.seed(1)
data = generate_covariance(20, 20)

# Compute weight matrix
W = cggm_weights(data$sample, phi = 1, k = 2)

# Set the target number of cluster to be the minimum achievable number of
# clusters based on the weight matrix
target = min_clusters(W)

# Set an initial sequence for lambdas
lambdas = seq(0, 1, 0.1)

# Compute solutions for the initial set of lambdas
result = cggm(S = data$sample, W = W, lambdas = lambdas, store_all_res = TRUE)

# While the target number of clusters has not been found, continue adding more
# solutions for larger values of lambda
while (min(result$cluster_counts) != target) {
    # Get the maximum value of lambda
    max_lambda = max(result$lambdas)
    if (max_lambda  == 0) max_lambda = 1

    # Set an additional sequence of values
    lambdas = seq(max_lambda, 2 * max_lambda, length.out = 5)[-1]

    # Compute additional results
    result = cggm_expand(cggm_output = result, lambdas = lambdas)
}

result$cluster_counts

# Compute difference in norms
diffnorms = rep(0, result$n - 1)
p = nrow(data$sample)

for (i in 2:result$n) {
    diffnorms[i - 1] = norm(result$Theta[[i - 1]] - result$Theta[[i]], "F") / p
}

# Plot versus lambdas
plot(result$lambdas[-1], diffnorms, type = "l")

## Now fill in the gaps where the difference between two consecutive solutions
## is too large. Do this by inserting values for lambda between those for which
## the consecutive solutions were too different.
# Maximum allowed difference between consecutive solutions
max_diff = 0.01

# Find the differences that exceed the maximum
indices = which(diffnorms > max_diff)

# Select lambdas
lambdas = c()

for (i in indices) {
    # Get minimum and maximum value of lambda
    min_lambda = result$lambdas[i]
    max_lambda = result$lambdas[i + 1]

    # Get the number of lambdas that should be inserted
    n_lambdas = floor(diffnorms[i] / max_diff)

    # Get a sequence that includes the minimum and maximum, and trim those
    lambdas_ins = seq(min_lambda, max_lambda, length.out = n_lambdas + 2)
    lambdas = c(lambdas, lambdas_ins[-c(1, n_lambdas + 2)])
}

# Compute additional results
result = cggm_expand(cggm_output = result, lambdas = lambdas)

# Compute difference in norms
diffnorms = rep(0, result$n - 1)
p = nrow(data$sample)

for (i in 2:result$n) {
    diffnorms[i - 1] = norm(result$Theta[[i - 1]] - result$Theta[[i]], "F") / p
}

# Plot versus lambdas
plot(result$lambdas[-1], diffnorms, type = "l")

# See how lambda progresses
plot(result$lambdas, type = "l")

## Fill in the gaps based on the numbers of clusters found by the sequence of
## values for lambda. This is done by inserting additional values between those
## for which the decrease in the number of clusters is too large
diff_clusters = abs(diff(result$cluster_counts))

# Find the differences larger than 1
indices = which(diff_clusters > 1)

# Select lambdas
lambdas = c()

for (i in indices) {
    # Get minimum and maximum value of lambda
    min_lambda = result$lambdas[i]
    max_lambda = result$lambdas[i + 1]

    # Get the number of lambdas that should be inserted
    n_lambdas = diff_clusters[i] - 1

    # Get a sequence that includes the minimum and maximum, and trim those
    lambdas_ins = seq(min_lambda, max_lambda, length.out = n_lambdas + 2)
    lambdas = c(lambdas, lambdas_ins[-c(1, n_lambdas + 2)])
}

# Compute additional results
result = cggm_expand(cggm_output = result, lambdas = lambdas)

# Check the number of clusters for each instance
result$cluster_counts

# See how lambda progresses
plot(result$lambdas, type = "l")
