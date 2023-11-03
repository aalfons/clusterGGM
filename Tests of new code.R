library(CGGMR)

# Generate some covariance data
set.seed(1)
data = generateCovariance(8, 5)
S = data$sample
R = solve(S)
A = diag(R)
diag(R) = 0
lambdas = c(1)

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

test(keys - 1, values, R, A, p, u - 1, S, lambdas)

res = CGGMR:::.cggm(Ri = R, Ai = A, pi = p, ui = u - 1, S = S, UWUi = UWU,
                    lambdas = lambdas, gss_tol = 1e-6, conv_tol = 1e-6,
                    fusion_check_threshold = 0, max_iter = 1, store_all_res = T,
                    fusion_type = 3, Newton_dd = T, print_profile_report = T,
                    verbose = 6)
res$losses
#cggm(S, UWU, lambdas, max_iter = 0)
