# Clear environment to prevent mistakes
rm(list = ls())
gc()

# Load packages
library(CGGMR)
library(igraph)
library(mclust)
library(viridis)

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
lambdas = seq(0, 0.50, 0.005)

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
res = cggm(S, W, lambdas, store_all_res = TRUE, verbose = 1)
res$cluster_counts
plot(res$lambdas, res$losses, type = "l", col = "black", lty = 1, lwd = 2,
     xlab = "lambda", ylab = "loss")

# The highest ARI is achieved for the solution with 11 clusters
index = res$cluster_solution_index[11]

# Adjusted Rand index
print(adjustedRandIndex(res$clusters[[index]], data$clusters))

# Refit without penalty but with clusters
refit_res = cggmRefit(res, S)

# The solution index with the highest ARI
refit_index = refit_res$cluster_solution_index[11]

# Mean squared deviation from the true Theta
mean((res$Theta[[index]] - solve(Sigma))^2)                # Fitted
mean((refit_res$Theta[[refit_index]] - solve(Sigma))^2)    # Refitted

# Let's take a look at the mean squared error versus the number of clusters
mses = c()
for (i in 1:res$n) {
    mse = mean((res$Theta[[i]] - solve(Sigma))^2)
    mses = c(mses, mse)
}

# Plot the MSE versus the number of clusters and color the points based on the
# value for lambda. The plot shows that for one solution with a fixed number of
# clusters, the lowest MSE is achieved by the smallest value for lambda. This
# adds to the intuition that if you want a particular number of clusters, you
# prefer the solution with the smallest possible lambda for that number
point_colors = viridis(1001, direction = -1)
point_colors = point_colors[round(res$lambdas / max(res$lambdas) * 1000) + 1]
plot(res$cluster_counts, mses, type = "l", col = "grey", lwd = 2,
     xlab = "Number of Clusters", ylab = "Mean Squared Error (MSE)",
     main = "MSE vs. Number of Clusters", cex.main = 1.2, cex.lab = 1.2)
points(res$cluster_counts, mses, pch = 16, col = point_colors, cex = 0.75)

# For completeness, we do a similar thing for the refitted solution, because
# this has some interesting results as well
mses = c()
for (i in 1:refit_res$n) {
    mse = mean((refit_res$Theta[[i]] - solve(Sigma))^2)
    mses = c(mses, mse)
}

# Add the lines to the previous plot, although somewhat hard to see, the minimum
# MSE for the refitted Theta is achieved for 11 clusters, lambda is always zero
# in this case
lines(refit_res$cluster_counts, mses, type = "l", col = "grey", lwd = 2)
points(refit_res$cluster_counts, mses, pch = 16, col = "black", cex = 0.75)
