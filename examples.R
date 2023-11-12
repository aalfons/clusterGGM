# Clear environment to prevent mistakes
rm(list = ls())
gc()

# Load packages
library(CGGMR)
library(igraph)


# Generate covariance matrix with a particular number of variables that are
# driven by an underlying cluster structure
set.seed(1)
data = generateCovariance(n_vars = 5, n_clusters = 2)

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
W = cggmWeights(S, phi = 1, k = 2)

# Plot the weight matrix as a weighted graph
G = graph_from_adjacency_matrix(W, mode = "undirected", weighted = TRUE)
plot(G, edge.label = round(E(G)$weight, 3), layout = layout.circle(G))

# Set lambda
lambdas = seq(0, 0.05, 0.005)

# Testing the algorithm, ?cggm provides some explanation of the inputs, here I
# discuss some important ones
# fusion_type: Currently can take on 4 values, of which "proximity" is the
#              default as it consistently outperforms the others during testing:
#              "proximity": no checks to fuse other than whether the distance
#                           d_{kl}(A, R) is smaller than  is smaller some small
#                           value.
#              "a0": when checking fusions, the row/column that is being
#                    minimized, denoted by index k, is set completely to the
#                    values in another row/column, denoted by index m. Then an
#                    analytical check is performed using subgradients whether
#                    the new situation is a minimum.
#              "a1": in this case, rows/cols k and m are set to the weighted
#                    average of the original rows/cols k and m. Then a check is
#                    only performed from the perspective of k: does the new
#                    situation minimize the loss function with respect to k.
#                    This ignores the loss with respect to m, and therefore is
#                    theoretically poorly motivated.
#              "a2": same as fusion_type = 1, but this time the check is done
#                    correct. For both k and m it is checked whether the loss
#                    with respect to k is minimized and whether the loss with
#                    respect to m is minimized using subgradients.
# fusion_threshold: For proximity based clustering, this is the threshold that
#                   determines fusions. For the analytical fusions it is used as
#                   an initial filter, as the check is computationally
#                   nontrivial, this can be used to only check fusions of
#                   variables that are sufficiently close.
# store_all_res: If true, the results for every value for lambda are stored, if
#                false, a result is only stored if the number of clusters has
#                decreased with respect to the result for the previous value for
#                lambda. The default is false. Plotting the obtained losses
#                against the lambdas can provide useful insights and serves as a
#                warning system if the results cannot be trusted.
res = cggm(S, W, lambdas, store_all_res = TRUE)

# New version of the algorithm, which performs minimization more efficiently
resNew = cggmNew(S, W, lambdas, store_all_res = TRUE)

# The result is a list of the following:
# losses: the values for the loss function obtained for the stored solutions
# lambdas: the values for lambda for which results were returned, if
#          store_all_res was true, this is the same as the input vector for
#          lambdas
# cluster_counts: the numbers of clusters for each of the solutions, if
#                 store_all_res was false, this should not contain duplicates
# Theta: all solutions for Theta
# R: all solutions for R
# A: all solutions for A
# clusters: all solutions for the cluster identifiers
# fusion_threshold: only present when proximity based clustering is used, the
#                   value used as threshold
# cluster_solution_index: a vector where the nth element denotes the index of
#                         the solution for which n clusters were found for the
#                         first time. For example, the value returned by
#                         cluster_solution_index[5] will return an index i that
#                         can be used to obtain the Theta with 5 clusters as
#                         Theta[[i]].
# n: the number of solutions for quick access
# Let's take a look at the cluster counts and plot the value of the loss against
# the lambdas
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
refit_res = cggmRefit(res, S)

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
# argument. This function still makes use of the less efficient code for the
# minimization.
lambdas = seq(0, 0.25, 0.01)
res_CV = cggmCV(X = data$data, lambdas = lambdas, phi = c(0.5, 1.5),
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
