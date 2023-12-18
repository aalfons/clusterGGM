# Clear environment to prevent mistakes
rm(list = ls())
gc()
par(mfrow = c(1, 1))

# Load packages
library(CGGMR)
library(igraph)


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
# for k in a "sensible" range: k in [1, nrow(S) - 1]). An optional argument is
# connected, which ensures a connected weight matrix
W = cggm_weights(S, phi = 1, k = 2)

# Plot the weight matrix as a weighted graph
G = graph_from_adjacency_matrix(W, mode = "undirected", weighted = TRUE)
plot(G, edge.label = round(E(G)$weight, 3), layout = layout.circle(G))

# Set lambda
lambdas = seq(0, 0.05, 0.005)

# Testing the algorithm, ?cggm provides some explanation of the inputs
res = cggm(S, W, lambda = lambdas)

# The result contains a lot of information. Here we take a look at the number of
# clusters found after minimizing each instance of the loss function
res$cluster_counts

# We also have accessor functions for both Theta and the cluster labels. Here
# the solution for the last value of lambda is retrieved
get_Theta(res, index = res$n)
get_clusters(res, index = res$n)

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
print(get_Theta(res, 15))               # Fitted
print(get_Theta(refit_res, 2))          # Refitted
print(solve(Sigma))                     # True

# We can see that, mostly due to the sparse weights and of course the easy toy
# example, there is only a small bias when comparing the fitted and refitted
# versions of Theta

# Perform k-fold CV to select the optimal value of phi, k, and lambda. This
# function is able to automatically tune the value of lambda if there is no
# column called lambda in the tune_grid data frame
res_cv = cggm_cv(
    X = data$data,
    tune_grid = expand.grid(phi = c(0.5, 1.5), k = c(1, 2, 3),
                            lambda = seq(0, 0.25, 0.01)),
    connected = TRUE # Is FALSE by default
)

# The optimal parameters
print(res_cv$opt_tune)

# The cluster labels after cross validation
print(get_clusters(res_cv))

# Theta after cross validation
print(get_Theta(res_cv))

# True Theta
print(solve(data$true))
res_cv$final$inputs
