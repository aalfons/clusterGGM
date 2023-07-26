# Clear environment to prevent mistakes
rm(list = ls())
gc()

# Load packages
library(CGGMR)
library(igraph)

# Generate covariance matrix with a particular number of variables that are
# driven by an underlying cluster structure
set.seed(1)
data = generateCovariance(n_vars = 50, n_clusters = 10)
Sigma = data$true
S = data$sample

# View true cluster labels
print(data$clusters)

# Compute weight matrix
W = cggmWeights(S, phi = 1, k = 5)

# Plot the weight matrix as a weighted graph
G = graph_from_adjacency_matrix(W, mode = "undirected", weighted = TRUE)
plot(G, edge.label = round(E(G)$weight, 3))

# Set lambda
lambdas = seq(0, 0.20, 0.005)

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
res = cggm(S, W, lambdas, store_all_res = TRUE, verbose = 1, profile = TRUE)
res$cluster_counts
plot(res$lambdas, res$losses, type = "l", col = "black", lty = 1, lwd = 2,
     xlab = "lambda", ylab = "loss")

# Index for solution with 10 clusters
index = res$cluster_solution_index[10]

# Refit without penalty but with clusters
refit_res = cggmRefit(res, S)

# The solution index with the correct number of clusters
refit_index = refit_res$cluster_solution_index[10]

# Mean absolute deviation from the true Theta
mean(abs(res$Theta[[index]] - solve(Sigma)))                # Fitted
mean(abs(refit_res$Theta[[refit_index]] - solve(Sigma)))    # Refitted
