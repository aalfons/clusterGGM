# Clear environment to prevent mistakes
rm(list = ls())
gc()

# Load packages
library(CGGMR)

# Generate covariance matrix
set.seed(1)
set.seed(10)
S = generateCovariance(n_vars = 15, n_clusters = 4)
Sigma = S$true
S = S$sample

# Initial estimate for Theta is the inverse of S
Theta = solve(S)
print(Theta)

# Compute weight matrix, k = 0 means a dense weight matrix
W = cggmWeights(Theta, phi = 1, k = 0)

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
lambdas = seq(0, 0.25, 0.01)

# Testing the algorithm with setting k to m in case of a fusion
res1 = cggm(S, W, lambdas, gss_tol = 1e-4, conv_tol = 1e-9,
            fusion_threshold = 1, max_iter = 1000, store_all_res = TRUE,
            verbose = 1, profile = TRUE, fusion_type = "a0", use_Newton = FALSE)
res1$cluster_counts
plot(res1$lambdas, res1$losses, type = "l", col = "black", lty = 1, lwd = 2)

# Testing the algorithm with setting k and m to the weighted mean of k and m.
# Test for fusion is done incorrectly: only from perspective of k
res2 = cggm(S, W, lambdas, gss_tol = 1e-4, conv_tol = 1e-9,
            fusion_threshold = 1, max_iter = 1000, store_all_res = TRUE,
            verbose = 1, profile = TRUE, fusion_type = "a1", use_Newton = FALSE)
res2$cluster_counts
lines(res2$lambdas, res2$losses, type = "l", col = "red", lty = 2, lwd = 2)

# Testing the algorithm with setting k and m to the weighted mean of k and m.
# Test for fusion is done correctly: from perspective of both k and m
res3 = cggm(S, W, lambdas, gss_tol = 1e-4, conv_tol = 1e-9,
            fusion_threshold = 1, max_iter = 1000, store_all_res = TRUE,
            verbose = 1, profile = TRUE, fusion_type = "a1", use_Newton = FALSE)
res3$cluster_counts
lines(res3$lambdas, res3$losses, type = "l", col = "blue", lty = 3, lwd = 2)

# Test optimizer
res4 = cggm2(S, W, lambdas)
lines(res4$lambdas, res4$losses, type = "l", col = "darkorange", lty = 4,
      lwd = 2)

# Testing the algorithm with setting k and m to the weighted mean of k and m.
# Test whether the variables are "close enough", no other testing is done
res5 = cggm(S, W, lambdas, gss_tol = 1e-4, conv_tol = 1e-9,
            fusion_threshold = 1e-5, max_iter = 1000, store_all_res = TRUE,
            verbose = 1, profile = TRUE, fusion_type = "proximity",
            use_Newton = TRUE)
lines(res5$lambdas, res5$losses, type = "l", col = "magenta", lty = 3, lwd = 2)
res5$cluster_counts

# Create plot
plot(res1$lambdas, res1$losses, type = "l", col = "black", lty = 1, lwd = 2)
lines(res4$lambdas, res4$losses, type = "l", col = "red", lty = 4,
      lwd = 2)
lines(res5$lambdas, res5$losses, type = "l", col = "blue", lty = 3, lwd = 2)
legend("bottomright", legend = c("Check Fusions", "Optim", "Naive Fusions"),
       col = c("black", "red", "blue"), lty = c(1, 4, 3), cex = 1)

# Testing existing implementation
#res4 = CGGM::cggm(Ri = R, Ai = A, pi = p, ui = u, S = S, UWUi = W,
#                  lambdas = lambdas, gss_tol = 1e-4, conv_tol = 1e-9,
#                  fusion_check_threshold = 1, max_iter = 1000,
#                  store_all_res = TRUE, verbose = 0)
#lines(res4$lambdas, res4$losses, type = "l", col = "green")
#res4$cluster_counts
