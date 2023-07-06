# Clear environment to prevent mistakes
rm(list = ls())
gc()

# Load packages
library(CGGMR)

# Generate covariance matrix
set.seed(1)
set.seed(12)
S = generateCovariance(n_vars = 7, n_clusters = 4)
Sigma = S$true
S = S$sample
Sigma

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
lambdas = seq(0, 0.10, 0.01)

# Testing the algorithm with setting k to m in case of a fusion
res1 = cggm(Ri = R, Ai = A, pi = p, ui = u, S = S, UWUi = W,
            lambdas = lambdas, gss_tol = 1e-4, conv_tol = 1e-9,
            fusion_check_threshold = 1, max_iter = 1000, store_all_res = TRUE,
            verbose = 1, print_profile_report = TRUE, fusion_type = 0)
plot(res1$lambdas, res1$losses, type = "l", col = "black", lty = 1)
res1$cluster_counts

# Convert output
res1 = convertCGGMOutput(res1)

# Get a solution
R = res1[[6]]$R
A = res1[[6]]$A
u = res1[[6]]$clusters
U = matrix(0, nrow = length(u), ncol = max(u))
U[cbind(1:length(u), u)] = 1
u = u - 1
p = apply(U, 2, sum)
UWU = t(U) %*% W %*% U

# Select variable
k = 2

# Set lambda
lambda = 1

# Small value
eps = 1e-4

# Compute inverse of R^*0
R_star_0_inv = computeRStar0Inv(R, A, p, k - 1)

# Gradient
gradient(R, A, p, u, R_star_0_inv, S, UWU, lambda, k - 1, -1)

# Matrix to store the Hessian in
H = matrix(0, nrow = nrow(R) + 1, ncol = ncol(R) + 1)

# Some necessary values
r_k = R[k, -k]
r_kk = R[k, k]
a_kk = A[k]

# Compute h()
h = (a_kk + (p[k] - 1) * r_kk - p[k] * t(r_k) %*% R_star_0_inv %*% r_k)[1, 1]

################################################################################
# For a_kk
################################################################################
# Numerical differentiation of the gradient wrt A[k]
A_ = A
A_[k] = A_[k] - eps
g1 = gradient(R, A_, p, u, R_star_0_inv, S, UWU, lambda, k - 1, -1)
A_[k] = A_[k] + 2 * eps
g2 = gradient(R, A_, p, u, R_star_0_inv, S, UWU, lambda, k - 1, -1)
(g2 - g1) / (2 * eps)

# LOG DET PART
# Second derivative wrt a_kk
H[1, 1] = 1 / h^2 + (p[k] - 1) / (a_kk - r_kk)^2

# Second derivative wrt r_k
H[1, -c(1, k + 1)] = -1 / h^2 * 2 * p[k] * R_star_0_inv %*% r_k
H[-c(1, k + 1), 1] = H[1, -c(1, k + 1)]

# Second derivative wrt r_kk
H[1, k + 1] =  (p[k] - 1) / h^2 - (p[k] - 1) / (a_kk - r_kk)^2
H[k + 1, 1] =  H[1, k + 1]

# CLUSTERPATH PART
# Second derivative wrt a_kk
for (l in 1:nrow(R)) {
    if (l != k) {
        d_kl = normRA(R, A, p, k - 1, l - 1)
        temp = UWU[k, l] * (1 / d_kl - (a_kk - A[l])^2 / d_kl^3)
        H[1, 1] = H[1, 1] + lambda * temp
    }
}

# Second derivative wrt r_kk
for (l in 1:nrow(R)) {
    if (l != k) {
        d_kl = normRA(R, A, p, k - 1, l - 1)
        temp = -UWU[k, l] * (r_kk - R[k, l]) * (a_kk - A[l]) / d_kl^3
        H[1, k + 1] = H[1, k + 1] + lambda * (p[k] - 1) * temp
    }
}
H[k + 1, 1] = H[1, k + 1]

# Second derivative wrt r_k
for (m in 1:nrow(R)) {
    if (m != k) {
        for (l in 1:nrow(R)) {
            if (l != k && l != m) {
                d_kl = normRA(R, A, p, k - 1, l - 1)
                temp = -UWU[k, l] * p[m] * (R[k, m] - R[m, l])
                temp = temp * (a_kk - A[l]) / d_kl^3
                H[1, m + 1] = H[1, m + 1] + lambda * temp
            }
        }

        d_km = normRA(R, A, p, k - 1, m - 1)
        temp = (p[k] - 1) * (R[k, m] - r_kk) + (p[m] - 1) * (R[k, m] - R[m, m])
        temp = -temp * UWU[k, m] * (a_kk - A[m]) / d_km^3
        H[1, m + 1] = H[1, m + 1] + lambda * temp

        H[m + 1, 1] = H[1, m + 1]
    }
}


################################################################################
# For r_kk
################################################################################
# Numerical differentiation of the gradient wrt R[k, k]
R_ = R
R_[k, k] = R_[k, k] - eps
g1 = gradient(R_, A, p, u, R_star_0_inv, S, UWU, lambda, k - 1, -1)
R_[k, k] = R_[k, k] + 2 * eps
g2 = gradient(R_, A, p, u, R_star_0_inv, S, UWU, lambda, k - 1, -1)
(g2 - g1) / (2 * eps)

# LOG DET PART
# Second derivative wrt r_kk
H[k + 1, k + 1] = (p[k] - 1)^2 / h^2 + (p[k] - 1) / (a_kk - r_kk)^2

# Second derivative wrt r_k
H[k + 1, -c(1, k + 1)] = -(p[k] - 1) / h^2 * 2 * p[k] * R_star_0_inv %*% r_k
H[-c(1, k + 1), k + 1] = H[k + 1, -c(1, k + 1)]

# CLUSTERPATH PART
# Second derivative wrt r_kk
for (l in 1:nrow(R)) {
    if (l != k) {
        d_kl = normRA(R, A, p, k - 1, l - 1)
        temp = UWU[k, l] * (1 / d_kl - (p[k] - 1) * (r_kk - R[k, l])^2 / d_kl^3)
        H[k + 1, k + 1] = H[k + 1, k + 1] + lambda * (p[k] - 1) * temp
    }
}

# Second derivative wrt r_k
for (m in 1:nrow(R)) {
    if (m != k) {
        for (l in 1:nrow(R)) {
            if (l != k && l != m) {
                d_kl = normRA(R, A, p, k - 1, l - 1)
                temp = -UWU[k, l] * p[m] * (p[k] - 1) * (r_kk - R[k, l])
                temp = temp * (R[k, m] - R[m, l]) / d_kl^3
                H[k + 1, m + 1] = H[k + 1, m + 1] + lambda * temp
            }
        }

        d_km = normRA(R, A, p, k - 1, m - 1)
        temp = (p[k] - 1) * (R[k, m] - r_kk) + (p[m] - 1) * (R[k, m] - R[m, m])
        temp = 1 - temp * (R[k, m] - r_kk) / d_km^2
        temp = -UWU[k, m] * (p[k] - 1) * temp / d_km

        H[k + 1, m + 1] = H[k + 1, m + 1] + lambda * temp
        H[m + 1, k + 1] = H[k + 1, m + 1]
    }
}

H
