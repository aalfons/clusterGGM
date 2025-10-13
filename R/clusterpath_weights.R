#' Compute the Weight Matrix for the Clusterpath Penalty
#'
#' Compute the (possibly sparse) weight matrix for the clusterpath penalty in
#' the clusterpath estimator of the Gaussian graphical model (CGGM). Weights
#' are computed based on a distance measure so that variables that are close
#' are clustered more quickly (higher weight) than variables that are far apart
#' (lower weight). Only neighboring variables thereby receive a nonzero weight.
#' Additionally, groups of variables that would not be connected via nonzero
#' weights due to the sparsity of the weight matrix can still be connected by
#' applying a minimum spanning tree algorithm.
#'
#' @param S The sample covariance matrix of the data.
#' @param phi Tuning parameter of the weights.
#' @param k The number of nearest neighbors that should be used to set weights
#' to a nonzero value. If \code{0 < k < ncol(S)}, the dense weight matrix will
#' be made sparse, otherwise the dense matrix is returned.
#' @param connected  A logical indicating whether a connected weight matrix
#' should be enforced. Defaults to \code{TRUE}.
#'
#' @return A weight matrix for the clusterpath penalty.
#'
#' @author Daniel J.W. Touw
#'
#' @references
#' D.J.W. Touw, A. Alfons, P.J.F. Groenen and I. Wilms (2025)
#' \emph{Clusterpath Gaussian Graphical Modeling}. arXiv:2407.00644.
#' doi:10.48550/arXiv.2407.00644.
#'
#' @seealso
#' \code{\link{lasso_weights}()}, \code{\link{cggm}()},
#' \code{\link{cggm_refit}()}, \code{\link{cggm_cv}()}
#'
#' @example inst/doc/examples/example-clusterpath_weights.R
#'
#' @importFrom Rcpp evalCpp
#' @export
clusterpath_weights <- function(S, phi, k, connected = TRUE)
{
    # Initial estimate for Theta
    Theta = .initial_Theta(S)

    # Get the squared norms for the distances between the variables in Theta
    sq_norms = .scaled_squared_norms(Theta)

    ## If k falls outside a reasonable range, return the dense weight matrix
    if (k <= 0 || k >= nrow(S)) {
        # Compute weights
        knn_weights = exp(-phi * sq_norms)

        # Set diagonal to zero
        diag(knn_weights) = 0

        return(knn_weights)
    }

    # Initialize the sparse weight matrix
    knn_weights = matrix(0, nrow = nrow(sq_norms), ncol = ncol(sq_norms))

    ## Fill a matrix with nonzero values according to the k-nn relation. The
    ## values are the weights between the variables. Also keep track of a matrix
    ## with the indices of the variables that are connected via nonzero weights,
    ## as this will be used later for ensuring connectedness.
    # Matrix to keep track of nonzero edges
    nz_edges = matrix(0, nrow = 2, ncol = k * nrow(Theta))
    nz_idx = 1

    for (i in 1:nrow(Theta)) {
        # Get the indices of the largest smallest distances, taking care to
        # ignore the distance of variable i to itself
        idx = .k_largest(-sq_norms[i, ], k)
        if (i %in% idx && k < nrow(Theta)) {
            idx = .k_largest(-sq_norms[i, ], k + 1)
            idx = idx[!idx %in% i]
        }

        for (j in idx) {
            # Store values in weight matrix
            knn_weights[i, j] = exp(-phi * sq_norms[i, j])
            knn_weights[j, i] = exp(-phi * sq_norms[i, j])

            # Keep track of nonzero edges in a different format
            nz_edges[1, nz_idx] = i
            nz_edges[2, nz_idx] = j

            nz_idx = nz_idx + 1
        }
    }

    # If connectedness is not important, return the weight matrix
    if (!connected) {
        return(knn_weights)
    }

    # Get the IDs of the different connected components that each variable
    # belongs to
    id = .find_subgraphs(nz_edges - 1, nrow(Theta)) + 1

    # Number of connected components
    n_clusters = max(id)

    # If the number of connected components is one, return the weight matrix
    if (n_clusters == 1) {
        return(knn_weights)
    }

    ## For each pair of connected components (or clusters), find the minimum
    ## distance between them and store the indices of the variables that are
    ## responsible for this distance.
    # List to keep track of which objects are responsible for those distances
    closest_objects = list()

    # Matrix to keep track of the smallest distance between the clusters
    dist_between = matrix(0, nrow = n_clusters, ncol = n_clusters)

    for (a in 2:n_clusters) {
        for (b in 1:(a - 1)) {
            # Select submatrix with between cluster distances
            partial_sq_norms = sq_norms[id == a, id == b]

            # Find the indices of the minimum value
            min_idx = arrayInd(which.min(partial_sq_norms),
                               dim(partial_sq_norms))

            # Get the indices of the objects with respect to their cluster
            a_idx = min_idx[1]
            b_idx = min_idx[2]

            # The minimum value
            min_val = partial_sq_norms[a_idx, b_idx]

            # Save the distance
            dist_between[a, b] = min_val
            dist_between[b, a] = min_val

            # Get the original indices of the objects
            a_idx = c(1:nrow(Theta))[id == a][a_idx]
            b_idx = c(1:nrow(Theta))[id == b][b_idx]

            # Store the objects
            idx = (a - 1) * (a - 2) / 2 + b
            closest_objects[[idx]] = c(a_idx, b_idx)
        }
    }

    ## dist_between holds the distances between the connected components, an MST
    ## algorithm can be applied to this distance matrix to find the edges that
    ## are required to fully connect the weight matrix.
    # Find minimum spanning tree for dist_between
    mst_edges = .find_mst(dist_between) + 1

    ## The matrix mst_edges contains the indices of the connected components,
    ## not the indices of the actual variables responsible for the distances in
    ## dist_between. The next step is to convert the indices of the connected
    ## components to the indices of the actual variables.
    for (i in 1:nrow(mst_edges)) {
        # Select a and b to convert the 2D index to a 1D index
        a = max(mst_edges[i, ])
        b = min(mst_edges[i, ])

        # Find the 1D index
        idx = (a - 1) * (a - 2) / 2 + b

        # Select the indices of the actual variables from the closest_objects
        # list
        mst_edges[i, ] = closest_objects[[idx]]
    }

    ## Finally, add the additional nonzero elements to the knn weight matrix
    for (i in 1:nrow(mst_edges)) {
        a = mst_edges[i, 1]
        b = mst_edges[i, 2]
        knn_weights[a, b] = exp(-phi * sq_norms[a, b])
        knn_weights[b, a] = exp(-phi * sq_norms[a, b])
    }

    return(knn_weights)
}
