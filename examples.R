# Clear environment to prevent mistakes
rm(list = ls())
gc()

# Load packages
library(CGGMR)
library(igraph)


## Example 1
# Generate covariance matrix with a particular number of variables that are
# driven by an underlying cluster structure
set.seed(1)
data = generate_covariance(n_vars = 5, n_clusters = 2)

# The variable data contains a true covariance matrix, which we store in Sigma,
# a sample covariance matrix S, and the true cluster labels
Sigma = data$true
S = data$sample

# View true Theta and true cluster labels
print(solve(Sigma))
print(data$clusters)

# Compute weight matrix, based on exp(-phi * d(Theta_i, Theta_j)), with sparsity
# based on the 2 nearest neighbors: k = 2 (the dense matrix is only made sparse
# for k in a "sensible" range: k in [1, nrow(S) - 1]). By default a connected
# weight matrix is guaranteed by the argument connected = TRUE
W = cggm_weights(S, phi = 1, k = 2)

# Plot the weight matrix as a weighted graph
G = graph_from_adjacency_matrix(W, mode = "undirected", weighted = TRUE)
plot(G, edge.label = round(E(G)$weight, 3), layout = layout.circle(G))

# Set lambda
lambdas = seq(0, 0.1, 0.01)

# Testing the algorithm, ?cggm provides some explanation of the inputs
res = cggm(S, W, lambda = lambdas)

# The result contains a lot of information. Here we take a look at the number of
# clusters found after minimizing each instance of the loss function
res$cluster_counts

# We also have accessor functions for both Theta and the cluster labels. Here
# the solution for the last value of lambda is retrieved
get_Theta(res, index = res$n)
get_clusters(res, index = res$n)


## Example 2
# Often, it is not clear which values of lambda make up a "sensible" sequence
# with appropriate step sizes from one to the next. That is why cggm() also has
# and expand argument, which automatically finds the smallest number of clusters
# and adds values of lambda so that the difference between consecutive solutions
# is not too large. Where "too large" is determined by the max_difference
# argument
res = cggm(S, W, lambda = lambdas, expand = TRUE)

# Finally, we can refit the result without penalty but with cluster constraints,
# this is not very relevant right now, but may be useful later if we want to
# compare the solution for Theta to the true value
refit_res = cggm_refit(res)

# View the results
print(get_Theta(res, 16))               # Fitted
print(get_Theta(refit_res, 3))          # Refitted
print(solve(Sigma))                     # True

# We can see that, mostly due to the sparse weights and of course the easy toy
# example, there is only a small bias when comparing the fitted and refitted
# versions of Theta


## Example 3
# Perform k-fold CV to select the optimal value of phi, k, and lambda. This
# function is able to automatically tune the value of lambda if there is no
# column called lambda in the tune_grid data frame
folds = cv_folds(nrow(data$data), 5)
res_cv = cggm_cv(
    X = data$data,
    tune_grid = expand.grid(phi = c(0.5, 1.5), k = c(1, 2, 3),
                            lambda = seq(0, 0.25, 0.01)),
    folds = folds,
    verbose = 1
)

# The optimal parameters
print(res_cv$opt_tune)

# The cluster labels after cross validation
print(get_clusters(res_cv))

# Theta after cross validation
print(get_Theta(res_cv))


## Example 4
# Perform k-fold CV with automatic lambda tuning. This does take a fair bit
# longer than providing a grid for lambda yourself
res_cv = cggm_cv(
    X = data$data,
    tune_grid = expand.grid(phi = c(0.5, 1.5), k = c(1, 2, 3)),
    folds = folds,
    verbose = 1
)

# The optimal parameters
print(res_cv$opt_tune)

# The cluster labels after cross validation
print(get_clusters(res_cv))

# Theta after cross validation
print(get_Theta(res_cv))


## Example 5
# Using cross validation to tune the parameters when refittig Theta is also
# possible. There is one issue: multiple settings for k and phi may yield the
# same clustering, and thus the same cross validation score. This makes choosing
# values for k and phi more difficult. To still be able to make a choice, the
# cross validation procedure takes note of the length of the interval for lambda
# that yields the lowest score for each combination of k and phi. To break ties,
# the k and phi with the largest interval for lambda are chosen. The optimal
# value for lambda is chosen as the midpoint of this interval
res_cv = cggm_cv(
    X = data$data,
    tune_grid = expand.grid(phi = c(0.5, 1.5), k = c(1, 2, 3)),
    folds = folds,
    refit = TRUE,
    verbose = 1
)

# The optimal parameters
print(res_cv$opt_tune)

# The cluster labels after cross validation
print(get_clusters(res_cv))

# Theta after cross validation
print(get_Theta(res_cv))


## Example 6
# Next, an example of a type of problem that distinguishes CGGM from other
# methods: one where the cluster structure is only apparent from the values on
# the diagonal of Theta.
Theta = matrix(c(2, 1, 1, 1,
                 1, 2, 1, 1,
                 1, 1, 4, 1,
                 1, 1, 1, 4),
               nrow = 4)
X = mvtnorm::rmvnorm(n = 200, sigma = solve(Theta))

# Apply cross validation
folds = cv_folds(nrow(X), 5)
res_cv = cggm_cv(
    X = X,
    tune_grid = expand.grid(phi = c(0.5, 1.25, 2.0), k = c(1, 2, 3)),
    folds = folds,
    verbose = 1
)

# Show the optimal tuning parameters
print(res_cv$opt_tune)

# The cluster labels after cross validation
print(get_clusters(res_cv))

# Theta after cross validation
print(get_Theta(res_cv))


## Example 7
# It is also possible to use the sample precision matrix as input to the
# optimization algorithm. For example, here we compute the weight matrix,
# compute a clusterpath, and refit the result while using solve(S) as input to
# estimate the covariance matrix instead of the precision matrix.
Theta_sample = solve(cov(X))

# Compute weight matrix
W = cggm_weights(Theta_sample, phi = 1, k = 2)

# Compute clusterpath
res_Sigma = cggm(Theta_sample, W, lambda = seq(0, 0.5, 0.05), expand = TRUE)

# Refit
refit_res_Sigma = cggm_refit(res_Sigma)

# The accessors still work
get_clusters(refit_res_Sigma, index = refit_res_Sigma$n - 1)
get_Theta(refit_res_Sigma, index = refit_res_Sigma$n - 1)

# The true covariance matrix
solve(Theta)


## Example 8
# Of course, we can also apply cross validation to get a similar result
res_Sigma_cv = cggm_cv(
    X = X,
    tune_grid = expand.grid(phi = c(0.5, 1.25, 2.0), k = c(1, 2, 3)),
    folds = folds,
    verbose = 1,
    refit = TRUE,
    estimate_Sigma = TRUE
)

# The cluster labels after cross validation
print(get_clusters(res_Sigma_cv))

# Theta after cross validation
print(get_Theta(res_Sigma_cv))

# The true covariance matrix
solve(Theta)
