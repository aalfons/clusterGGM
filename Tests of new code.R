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

test(keys - 1, values, R, A, p, u - 1, S, lambdas, 1)

res = CGGMR:::.cggm(Ri = R, Ai = A, pi = p, ui = u - 1, S = S, UWUi = UWU,
                    lambdas = lambdas, gss_tol = 1e-6, conv_tol = 1e-6,
                    fusion_check_threshold = 0, max_iter = 1, store_all_res = T,
                    fusion_type = 3, Newton_dd = T, print_profile_report = F,
                    verbose = 4)
res$losses
#cggm(S, UWU, lambdas, max_iter = 0)

k = 1
D = matrix(nrow = ncol(R), ncol = ncol(R))
for (i in 1:ncol(R)) {
    for (j in 1:ncol(R)) {
        D[i, j] = normRA(R, A, p, i - 1, j - 1)
    }
}

D2 = D^2
for (i in 1:ncol(R)) {
    for (j in 1:ncol(R)) {
        D2[i, j] = D2[i, j] - p[k] * (R[i, k] - R[j, k])^2
    }
}

R[k, -k] = R[k, -k] + 0.1
R[-k, k] = R[-k, k] + 0.1
A[k] = A[k] + 0.1

E = matrix(nrow = ncol(R), ncol = ncol(R))
for (i in 1:ncol(R)) {
    for (j in 1:ncol(R)) {
        E[i, j] = normRA(R, A, p, i - 1, j - 1)
    }
}

E2 = D2
for (i in 1:ncol(R)) {
    for (j in 1:ncol(R)) {
        E2[i, j] = E2[i, j] + p[k] * (R[i, k] - R[j, k])^2
    }
}
E2 = sqrt(E2)
