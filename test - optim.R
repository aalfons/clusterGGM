# Clear environment to prevent mistakes
rm(list = ls())
gc()

# Load packages
library(CGGMR)

# Generate covariance matrix
set.seed(10)
set.seed(1)
S = generateCovariance(n_vars = 15, n_clusters = 4)
Sigma = S$true
S = S$sample

# Initial estimate for Theta is the inverse of S
Theta = solve(S)
print(Theta)

# Compute weight matrix
W = weightsTheta(Theta, 1)

# Initialize R, A, u
# R and A are self explanatory
# u is the cluster membership vector
R = Theta
diag(R) = 0
A = diag(Theta)
u = c(1:ncol(S))

# Compute vector with cluster sizes
p = rep(0, max(u))
for (i in 1:length(u)) {
    p[u[i]] = p[u[i]] + 1
}

# Ensure indices start at 0, silly R
u = u - 1

# Cleanup
rm(i)

# Set lambda
lambdas = seq(0, 0.15, 0.01)

# Testing the algorithm with setting k to m in case of a fusion
res1 = cggm(Ri = R, Ai = A, pi = p, ui = u, S = S, UWUi = W,
            lambdas = lambdas, gss_tol = 1e-4, conv_tol = 1e-9,
            fusion_check_threshold = 1, max_iter = 10000, store_all_res = TRUE,
            verbose = 1, print_profile_report = TRUE, fusion_type = 0,
            Newton_dd = TRUE)
plot(res1$lambdas, res1$losses, type = "l", col = "black", lty = 1)
res1$cluster_counts


# Use optimizer
res2 = cggm2(S, W, lambdas)
lines(res2$lambdas, res2$losses, type = "l", col = "red", lty = 2)

