# Clear environment to prevent mistakes
rm(list = ls())
gc()

# Load packages
library(CGGMR)


# Generate covariance matrix
set.seed(1)
data = generate_covariance(n_vars = 5, n_clusters = 2)
S = data$sample

# Weights
W = cggm_weights(S, phi = 2, k = 2, connected = TRUE)

# Set lambda
lambdas = seq(0, 0.05, 0.01)

# Minimize
res = cggm(S, W, lambda = lambdas, expand = TRUE)

################################################################################
# MDS version
################################################################################
mod_sign <- function(vec)
{
    for (k in 2:length(vec)) {
        if (abs(vec[k] - vec[k - 1]) >
            abs(-vec[k] - vec[k - 1]))
        {
            vec[k] = -vec[k]
        }
    }

    return(vec)
}

i = 3
test = CGGMR:::.scaled_squared_norms(res$Theta[[i]])^0.5
test = cmdscale(test)
test

n = max(which(res$cluster_counts > 1))

mds_list = list()
for (i in 1:n) {
    test = CGGMR:::.squared_norms(res$Theta[[i]])^0.5
    mds_list[[i]] = cmdscale(test)
}

# Coordinates of the clusterpath
coords = do.call(rbind, mds_list)

# Number of variables
p = nrow(S)

# Plot start locations
plot(coords[1:p, ])

# Plot trails
for (j in 1:p) {
    idx_j = seq(j, p * n, p)
    coords_x_j = coords[idx_j, 1]
    coords_y_j = coords[idx_j, 2]

    coords_x_j = mod_sign(coords_x_j)
    coords_y_j = mod_sign(coords_y_j)

    lines(coords_x_j, coords_y_j)
}

