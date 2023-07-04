# Clear environment to prevent mistakes
rm(list = ls())
gc()

# Load packages
library(CGGM)

# Generate covariance matrix
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
lambdas = seq(0, 0.25, 0.01)

# Testing the algorithm
res = cggm(Ri = R, Ai = A, pi = p, ui = u, S = S, UWUi = W,
           lambdas = lambdas, gss_tol = 1e-4, conv_tol = 1e-9,
           fusion_check_threshold = 1, max_iter = 1000, store_all_res = TRUE,
           verbose = 1)
plot(res$lambdas, res$losses, type = "l")
res$cluster_counts

# Set lambda
lambdas = seq(0, 0.25, 0.005)
lambdas = seq(0, 0.13, 0.005)

# Testing the algorithm, only difference is the number of maximum iterations
res2 = cggm(Ri = R, Ai = A, pi = p, ui = u, S = S, UWUi = W,
            lambdas = lambdas, gss_tol = 1e-5, conv_tol = 1e-8,
            fusion_check_threshold = 1, max_iter = 100, store_all_res = TRUE,
            verbose = 1)
lines(res2$lambdas, res2$losses, type = "l")

# Convert result into more readable stuff
res3 = convertCGGMOutput(res2)
res3

# Do some tests with minimization
sol = 26
lambdas = c(res3[[sol]]$lambda)

res3[[sol]]

# Select a starting point for the minimization
R_sol = res3[[sol]]$R
A_sol = res3[[sol]]$A
u_sol = res3[[sol]]$clusters
U_sol = matrix(0, nrow = length(u_sol), ncol = max(u_sol))
U_sol[cbind(1:length(u_sol), u_sol)] = 1
u_sol = u_sol - 1
p_sol = apply(U_sol, 2, sum)
UWU_sol = t(U_sol) %*% W %*% U_sol
rm(U_sol)

res4 = cggm(Ri = R_sol, Ai = A_sol, pi = p_sol, ui = u_sol, S = S,
            UWUi = UWU_sol, lambdas = lambdas, gss_tol = 1e-5, conv_tol = 1e-8,
            fusion_check_threshold = 1, max_iter = 1000, store_all_res = TRUE,
            verbose = 3)

# The next bit of code is for when the DGP is called with n_vars = 5 and for the
# minimization lambdas = seq(0, 1, 0.01) is used

# Minimize the loss with clustered input and lambda=0
R = res[[2]]$R
A = res[[2]]$A
u = res[[2]]$clusters
U = matrix(0, nrow = length(u), ncol = max(u))
U[cbind(1:length(u), u)] = 1
u = u - 1
p = apply(U, 2, sum)
UWU = t(U) %*% W %*% U

res2 = cggm(Ri = R, Ai = A, pi = p, ui = u, S = S, UWUi = UWU,
            lambdas = c(0), gss_tol = 1e-4, conv_tol = 1e-8,
            fusion_check_threshold = 1e-3, max_iter = 1000, store_all_res = FALSE,
            verbose = 1)
res2 = convertCGGMOutput(res2)

# Compare clustered Theta with lambda =/= 0 with sample Theta
round(res[[2]]$Theta, 2)
round(Theta, 2)
round(res[[2]]$Theta - Theta, 2)

# Compare clustered Theta with lambda = 0 with sample Theta
round(res2[[1]]$Theta, 2)
round(Theta, 2)
round(res2[[1]]$Theta - Theta, 2)

# Compare clustered Theta with lambda = 0 with true Theta
round(res2[[1]]$Theta, 2)
round(solve(Sigma), 2)
round(res2[[1]]$Theta - solve(Sigma), 2)
