library(CGGMR)


# Construct a sparse matrix with key value pairs
keys = matrix(c(0, 1,
                0, 2,
                1, 3,
                2, 4,
                2, 3), nrow = 2, byrow = FALSE)
values = c(1, 3, 2, 9, 4)
keys = cbind(keys, keys[c(2, 1), ])
values = c(values, values)

# Create a dense version of the matrix
M = matrix(0, nrow = 5, ncol = 5)
for (i in 1:ncol(keys)) {
    M[keys[1, i] + 1, keys[2, i] + 1] = values[i]
}

# Create a membership vector to test fusions of the rows/columns
u = c(1, 2, 3, 1, 1)

# Check on the dense version
U = matrix(0, nrow = length(u), ncol = max(u))
U[cbind(1:length(u), u)] = 1
t(U) %*% M %*% U

# Generate some covariance data
set.seed(1)
data = generateCovariance(5, 5)
S = data$sample
R = solve(S)
A = diag(R)
diag(R) = 0
p = rep(1, nrow(R))
u = seq(1, nrow(R))
lambdas = c(0)

# Compute weights in dense format
UWU = cggmWeights(S, 2, 2)

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
                    fusion_check_threshold = 1, max_iter = 0, store_all_res = T,
                    fusion_type = 3, Newton_dd = T, print_profile_report = T,
                    verbose = 4)
res$losses
#cggm(S, UWU, lambdas, max_iter = 0)
