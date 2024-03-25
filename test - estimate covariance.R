# Clear environment to prevent mistakes
rm(list = ls())
gc()

# Load packages
library(CGGMR)


# Generate covariance matrix
set.seed(1)
data = generate_covariance(n_vars = 5, n_clusters = 2)

# The variable data contains a true covariance matrix, which we store in Sigma,
# a sample covariance matrix S, and the true cluster labels
Sigma = data$true
S = data$sample
Theta = solve(data$sample)

# View true cluster labels
print(data$clusters)

# Compute weight matrix for model that estimates Theta based on S
W_S = cggm_weights(S, phi = 1, k = 2)

# Compute weight matrix for model that estimates S based on Theta
W_Theta = cggm_weights(Theta, phi = 1, k = 2)

# Set lambda
lambdas = seq(0, 0.1, 0.01)

# Model with S as input
res_S = cggm(S, W_S, lambda = lambdas, expand = TRUE)
res_S = cggm_refit(res_S)
res_S$clusters

# Model with Theta as input
res_Theta = cggm(Theta, W_Theta, lambda = lambdas, expand = TRUE)
res_Theta = cggm_refit(res_Theta)
res_Theta$clusters

# Get estimates
S_hat = get_Theta(res_Theta, res_Theta$cluster_solution_index[2])
Theta_hat = get_Theta(res_S, res_S$cluster_solution_index[2])

# Difference with true precision matrix
Theta_hat - solve(data$true)

# Difference with true covariance matrix
S_hat - data$true

# Difference between estimates
S_hat - solve(Theta_hat)
solve(S_hat) - Theta_hat
