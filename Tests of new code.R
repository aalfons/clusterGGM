library(CGGMR)

# Generate some covariance data
set.seed(1)
data = generateCovariance(8, 5)
S = data$sample
R = solve(S)
A = diag(R)
diag(R) = 0
lambdas = c(0.1)

# Create a membership vector to test fusions of the rows/columns
u = data$clusters
U = matrix(0, nrow = length(u), ncol = max(u))
U[cbind(1:length(u), u)] = 1
p = apply(U, 2, sum)

# Reduce A and R
A = A[!duplicated(u)]
R = R[!duplicated(u), !duplicated(u)]

# Compute weights in dense format
UWU = cggmWeights(S, 2, 2)
UWU = t(U) %*% UWU %*% U
diag(UWU) = 0

# Transform weights into sparse format
keys = matrix(nrow = 2, ncol = 0)
values = c()
for (i in 1:nrow(UWU)) {
    for (j in 1:ncol(UWU)) {
        if (UWU[i, j] == 0) {
            next
        }

        keys = cbind(keys, matrix(c(i, j), nrow = 2))
        values = c(values, UWU[i, j])
    }
}

D = matrix(0, nrow = nrow(R), ncol = ncol(R))
for (i in 1:nrow(R)) {
    for (j in 1:ncol(R)) {
        D[i, j] = normRA(R, A, p, i - 1, j - 1)
    }
}

res1 = CGGMR:::.cggm2(keys - 1, values, R, A, p, u - 1, S, lambdas, 1,
                      conv_tol = 1e-6, max_iter = 1, store_all_res = T, verbose = 0)

res2 = CGGMR:::.cggm(Ri = R, Ai = A, pi = p, ui = u - 1, S = S, UWUi = UWU,
                     lambdas = lambdas, gss_tol = 1e-6, conv_tol = 1e-6,
                     fusion_check_threshold = 1, max_iter = 1, store_all_res = T,
                     fusion_type = 3, Newton_dd = T, print_profile_report = F,
                     verbose = 6)
res$losses


################################################################################
#
################################################################################
rm(list = ls())
gc()
library(CGGMR)

# Generate some covariance data
set.seed(1)
data = generateCovariance(7, 6)

# Covariance matrix
S = data$sample

# View true cluster labels
print(data$clusters)

# Compute weight matrix
W = cggmWeights(S, phi = 1, k = 2)

lambdas = seq(0, 0.10, 0.01)

res1 = cggmNew(S = S, W = W, lambdas = lambdas, gss_tol = 1e-4, conv_tol = 1e-7,
               fusion_threshold = NULL, max_iter = 100, store_all_res = T,
               verbose = 0)

res2 = cggm(S, W, lambdas, gss_tol = 1e-4, conv_tol = 1e-7, max_iter = 100,
            store_all_res = TRUE, use_Newton = TRUE, profile = TRUE,
            verbose = 0)
res1$losses
res2$losses

max(res1$losses / res2$losses)
min(res1$losses / res2$losses)
plot(res1$losses / res2$losses)

res1$cluster_counts
res2$cluster_counts

res1$R[[18]] - res2$R[[18]]

res1$clusters[[5]]
res2$clusters[[5]]

res2$R[[17]]
res2$A[[17]]


################################################################################
#
################################################################################
rm(list = ls())
gc()
library(CGGMR)

# Generate some covariance data
set.seed(1)
data = generateCovariance(7, 6)

# Covariance matrix
S = data$sample

# View true cluster labels
print(data$clusters)

# Compute weight matrix
W = cggmWeights(S, phi = 1, k = 2)

# Get the minimum number of clusters based on the weight matrix
c_min = CGGMR::min_clusters(W)

inital_lambdas = seq(0, 1, 0.05)

# Compute solutions for the initial set of lambdas, storing only the results for
# the smallest value for lambda for which a new number of cluster was attained
result = cggmNew(S = S, W = W, lambdas = inital_lambdas, gss_tol = 1e-4, conv_tol = 1e-7,
               fusion_threshold = NULL, max_iter = 100, store_all_res = TRUE,
               verbose = 0)

result$cluster_counts[result$n]

# Get a range for the additional values for lambda that will be tried
lambda_l = result$lambdas[length(result$lambdas)] * 1.01
lambda_u = 2 * lambda_l / 1.01
lambdas = seq(lambda_l, lambda_u, length.out = 20)




















# The steps down in the number of clusters
delta_clusters = diff(result$cluster_counts)

# Select the first index for which the decrease in the number of clusters is
# more than one
idx = which(delta_clusters < -1)[1]

# Between these two values of lambda lies at least one missing number of
# clusters
lambda_l = result$lambdas[idx]
lambda_u = result$lambdas[idx + 1]

# Select a range of lambdas to insert at that position
(lambda_u - lambda_l) * 0.05
(lambda_u - lambda_l) * 0.05


selected_lambdas = res1$lambdas
