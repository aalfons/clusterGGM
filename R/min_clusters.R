#' Calculate the Minimum Number of Clusters
#'
#' Compute the minimum number of clusters achievable using the provided weight
#' matrix.
#'
#' @param W The weight matrix used in the clusterpath penalty.
#'
#' @return An integer representing the minimum number of clusters.
#'
#' @export
min_clusters <- function(W)
{
    # Change later, but for now transform the dense matrix into a sparse one to
    # obtain the indices of nonzero elements
    # Numer of nonzero elements
    nnz = 2 * sum(W[lower.tri(W)] > 0)

    # Keys and values
    W_keys = matrix(nrow = 2, ncol = nnz)
    W_values = rep(0, nnz)

    # Fill keys and values
    idx = 1
    for (j in 1:ncol(W)) {
        for (i in 1:nrow(W)) {
            if (W[i, j] <= 0) next

            # Fill in keys and values
            W_keys[1, idx] = i - 1
            W_keys[2, idx] = j - 1
            W_values[idx] = W[i, j]
            idx = idx + 1
        }
    }

    n = max(W_keys) + 1

    return(CGGMR:::.count_clusters(W_keys, n))
}
