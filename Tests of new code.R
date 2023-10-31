library(CGGMR)


M = matrix(0, nrow = 5, ncol = 5)
keys = matrix(c(0, 1,
                0, 2,
                1, 3,
                2, 4,
                2, 3), nrow = 2, byrow = FALSE)
values = c(1, 3, 2, 9, 4)

for (i in 1:ncol(keys)) {
    M[keys[1, i] + 1, keys[2, i] + 1] = values[i]
}
M = M + t(M)

u = c(1, 2, 3, 4, 4)
U = matrix(0, nrow = length(u), ncol = max(u))
U[cbind(1:length(u), u)] = 1

t(U) %*% M %*% U

test(keys, values, nrow(M), u - 1)
