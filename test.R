library(CGGMR)

A = diag(4)
A = matrix(rnorm(4 * 4), nrow = 4)
a = 1:4
i = 2

A2 = A
A2[i, ] = A2[i, ] + a
solve(A2)

A2[, i] = A2[, i] + a
solve(A2)

updateInverse(solve(A), a, i - 1)
