# Clear environment to prevent mistakes
rm(list = ls())
gc()

# Load package
library(CGGMR)

# Generate covariance matrix with a particular number of variables that are
# driven by an underlying cluster structure
set.seed(1)
data = generateCovariance(n_vars = 5, n_clusters = 4)

# The variable data contains a true covariance matrix, which we store in Sigma,
# and a sample covariance matrix S
Sigma = data$true
S = data$sample
rm(data)

# Initial estimate for Theta is the inverse of S
Theta = solve(S)
print(Theta)

# Compute weight matrix, based on exp(-phi * d(Theta_i, Theta_j))
W = weightsTheta(Theta, phi = 1.0)

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
rm(i)

# Ensure indices start at 0, as C++ expects it this way
u = u - 1

# Set lambda
lambdas = seq(0, 0.14, 0.01)

# Testing the algorithm, the inputs are as follows
# Ri: R, the i is there to make naming inside the C++ function easier
# Ai: A
# pi: p, cluster sizes
# ui: u, membership vector
# UWUi: the potentially clustered version of W, computed as U^T * W * U
# lambdas: lambdas, make sure the values are monotonically increasing
# gss_tol: tolerance for the golden section search to determine the step size
#          during gradient descent
# conv_tol: tolerance to determine convergence of the minimization for a single
#           value for lambda
# fusion_check_threshold: if the distance d_{kl}(A, R) is smaller than
#                         fusion_check_threshold * sqrt(nrow(S)), the algorithm
#                         will perform a check whether a fusion should occur.
#                         behavior changes slightly depending on the value of
#                         fusion_type
# max_iter: maximum number of iterations while minimizing for a single value for
#           lambda
# store_all_res: if true, the results for every value for lambda are stored, if
#                false, a result is only stored if the number of clusters has
#                decreased with respect to the result for the previous value for
#                lambda
# verbose: determines the amount of information printed while the algorithm is
#          running. The highest level (verbose > 2) will perform additional
#          computations and slow down the algorithm
# print_profile_report: if true, prints a basic profiling report of the code
# fusion_type: currently can take on 4 values:
#              0: when checking fusions, the row/column that is being minimized,
#                 denoted by index k, is set completely to the values in another
#                 row/column, denoted by index m. Then an analytical check is
#                 performed using subgradients whether the new situation is a
#                 minimum
#              1: in this case, rows/cols k and m are set to the weighted
#                 average of the original rows/cols k and m. Then a check is
#                 only performed from the perspective of k: does the new
#                 situation minimize the loss function with respect to k. This
#                 ignores the loss with respect to m, and therefore is
#                 theoretically poorly motivated
#              2: same as fusion_type = 1, but this time the check is done
#                 correct. For both k and m it is checked whether the loss with
#                 respect to k is minimized and whether the loss with respect to
#                 m is minimized using subgradients
#              3: no checks to fuse other than whether the distance d_{kl}(A, R)
#                 is smaller than  is smaller than
#                 fusion_check_threshold * sqrt(nrow(S)), in this case it is
#                 important to choose fusion_check_threshold appropriately, for
#                 example to 1e-3
# Newton_dd: boolean to indicate whether a Newton descent direction should be
#            used instead of the gradient during the gradient descent part of
#            the algorthm. Note that this requires the computation of the
#            Hessian and its inverse
res = CGGMR:::.cggm(Ri = R, Ai = A, pi = p, ui = u, S = S, UWUi = W,
                    lambdas = lambdas, gss_tol = 1e-4, conv_tol = 1e-8,
                    fusion_check_threshold = 10, max_iter = 1000,
                    store_all_res = TRUE, verbose = 1,
                    print_profile_report = TRUE, fusion_type = 0,
                    Newton_dd = TRUE)
res$cluster_counts
# The result is a list of the following:
# clusters: a matrix with a column for each value for lambda and a row for each
#           of the variables in S. Each column contains the cluster ID for each
#           variable
# R: all solutions for R stacked horizontally, padded with zeros vertically if
#    the number of clusters (and hence the number of rows) decreases
# A: all solutions for A stacked horizontally, also padded with zeros vertically
#    if the number of clusters decreases
# lambdas: the values for lambda for which results were returned, if
#          store_all_res was true, this is the same as the input vector for
#          lambdas
# losses: the values for the loss function obtained for the stored solutions
# cluster_counts: the numbers of clusters for each of the solutions, if
#                 store_all_res was false, this should not contain duplicates
# Let's take a look at the cluster counts and plot the value of the loss against
# the lambdas
res$cluster_counts
plot(res$lambdas, res$losses, type = "l", col = "black", lty = 1, lwd = 2,
     xlab = "lambda", ylab = "loss")

# Because the result is difficult to browse, we can convert it into something
# easier to view
res = convertCGGMOutput(res)

# Each index of res contains the solution for one of the values for lambda,
# let's take a look at the first solution with four clusters. It consists of
# several elements: A, R, Theta, clusters, lambda, loss. Quite self explanatory,
# except maybe clusters, this is now a vector with cluster IDs. We can see that
# the clustered variables are variables 1 and 2
res[[11]]$Theta
res[[11]]$clusters

# Our solution is actually quite different from the true value for Theta
solve(Sigma)

# So we can also minimize the negative log likelihood subject to a clustering
# constraint. We do this by using a clustered solution as warm start and setting
# lambdas to c(0)
R = res[[11]]$R
A = res[[11]]$A
u = res[[11]]$clusters
U = matrix(0, nrow = length(u), ncol = max(u))
U[cbind(1:length(u), u)] = 1
u = u - 1
p = apply(U, 2, sum)
UWU = t(U) %*% W %*% U
rm(U)

res = cggm(Ri = R, Ai = A, pi = p, ui = u, S = S, UWUi = UWU,
           lambdas = c(0), gss_tol = 1e-4, conv_tol = 1e-9,
           fusion_check_threshold = 1, max_iter = 1000, store_all_res = TRUE,
           verbose = 1, print_profile_report = TRUE, fusion_type = 0,
           Newton_dd = FALSE)
res = convertCGGMOutput(res)

# Now the estimated Theta and true Theta are much more similar
res[[1]]$Theta
solve(Sigma)

# And for good measure: this is the sample Theta
solve(S)
