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
# for k in a "sensible" range: k in [1, nrow(S) - 1])
W = cggm_weights(S, phi = 1, k = 2)

# Plot the weight matrix as a weighted graph
G = graph_from_adjacency_matrix(W, mode = "undirected", weighted = TRUE)
plot(G, edge.label = round(E(G)$weight, 3), layout = layout.circle(G))

# Set lambda
lambdas = seq(0, 0.05, 0.005)

# Testing the algorithm, ?cggm provides some explanation of the inputs
res = cggm(S, W, lambdas, store_all_res = TRUE)

# The result contains a lot of information, here I showcase some of it. First we
# have the progression of the number of clusters for the values of lambda
res$cluster_counts

# We can plot the values of the loss function versus the lambdas
plot(res$lambdas, res$losses, type = "l", col = "black", lty = 1, lwd = 2,
     xlab = "lambda", ylab = "loss")

# If additional minimizations are required, it is possible to expand the result.
# This function takes as input the result from a call to cggm() and a vector
# for lambda, it then inserts solutions for the new values of lambda. It uses
# warm starts based on the supplied input, which makes it an efficient way to
# expand the solutionpath
res = cggm_expand(res, lambdas = seq(0.05, 0.10, 0.01))
res$cluster_counts
plot(res$lambdas, res$losses, type = "l", col = "black", lty = 1, lwd = 2,
     xlab = "lambda", ylab = "loss")

# Let's take the graph from before and color the nodes based on a clustering, we
# choose the solution based on the cluster_solution_index
index = res$cluster_solution_index[2]
V(G)$color = c("red", "blue")[res$clusters[[index]]]

# Make two plots side by side, the left one is colored based on our solution,
# the right one is colored based on the true clustering
og = par(mfrow = c(1, 2))
plot(G, edge.label = round(E(G)$weight, 3), layout = layout.circle(G))

# The data object also contains the true cluster labels, which can be used to
# color the nodes as well
V(G)$color = c("red", "blue")[data$clusters]
plot(G, edge.label = round(E(G)$weight, 3), layout = layout.circle(G))

# Revert modifications to par()
par(og)

# Finally, we can refit the result without penalty but with cluster constraints,
# this is not very relevant right now, but may be useful later if we want to
# compare the solution for Theta to the true value
refit_res = cggm_refit(res)

# The solution index with the correct number of clusters
refit_index = refit_res$cluster_solution_index[2]

# View the results
print(res$Theta[[index]])               # Fitted
print(refit_res$Theta[[refit_index]])   # Refitted
print(solve(Sigma))                     # True

# We can see that, mostly due to the sparse weights and of course the easy toy
# example, there is only a small bias when comparing the fitted and refitted
# versions of Theta

# Perform k-fold CV to select the optimal value of phi, k, and lambda. Among the
# items returned is an array with the scores for each value of lambda and also
# the result of the model after using the optimal values in the minimization.
# The cross validation function also accepts user-defined folds via the folds
# argument. This function makes use of the new code (cggmNew) for the
# minimizations.
lambdas = seq(0, 0.25, 0.01)
res_CV = cggm_cv(X = data$data, lambdas = lambdas, phi = c(0.5, 1.5),
                 k = c(1, 2, 3), kfold = 5)

# Plot the cross validation results
ylim = range(res_CV$scores)
ylim[2] = res_CV$scores[1, 1, 1]
plot(lambdas, res_CV$scores[, 1, 1], type = "l", xlab = "lambda",
     ylab = "CV score", col = "blue", ylim = ylim)
lines(lambdas, res_CV$scores[, 1, 2], col = "red")
lines(lambdas, res_CV$scores[, 1, 3], col = "green")
lines(lambdas, res_CV$scores[, 2, 1], col = "purple")
lines(lambdas, res_CV$scores[, 2, 2], col = "orange")
lines(lambdas, res_CV$scores[, 2, 3], col = "pink")

# Add legend
legend(
    "topright",
     legend = c("(phi, k)", "(0.5, 1)", "(0.5, 2)", "(0.5, 3)", "(1.5, 1)",
                "(1.5, 2)", "(1.5, 3)"),
     col = c("white", "blue", "red", "green", "purple", "orange", "pink"),
     lwd = 1
)

# The optimal parameters
print(res_CV$phi)
print(res_CV$k)
print(res_CV$lambda)

# The cluster labels after cross validation
print(res_CV$clusters)

# The clustered precision matrix
print(res_CV$Theta)

# Demonstrating the min_clusters function. First, generate some new data and
# compute a sparse weight matrix.
data = generate_covariance(n_vars = 10, n_clusters = 6)
Sigma = data$true
S = data$sample
W = cggm_weights(S, phi = 1, k = 1)

# What is the minimum number of clusters?
min_clusters(W)

# Plot the weight matrix as a weighted graph to check whether we see the same
# number of connected components
G = graph_from_adjacency_matrix(W, mode = "undirected", weighted = TRUE)
plot(G, edge.label = round(E(G)$weight, 3), layout = layout.circle(G))
