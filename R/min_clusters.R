#' Calculate the Minimum Number of Clusters
#'
#' Compute the minimum number of clusters achievable by the clusterpath penalty
#' using the provided weight matrix.
#'
#' @param W The weight matrix for the clusterpath penalty.
#'
#' @return An integer giving the minimum number of clusters.
#'
#' @author Daniel J.W. Touw
#'
#' @references
#' D.J.W. Touw, A. Alfons, P.J.F. Groenen and I. Wilms (2025)
#' \emph{Clusterpath Gaussian Graphical Modeling}. arXiv:2407.00644.
#' doi:10.48550/arXiv.2407.00644.
#'
#' @seealso
#' \code{\link{clusterpath_weights}()}, \code{\link{cggm}()},
#' \code{\link{cggm_cv}()}
#'
#' @example inst/doc/examples/example-min_clusters.R
#'
#' @export
min_clusters <- function(W)
{
    # Indices of the nonzero elements, which are the edges of the graph
    W_keys = .convert_to_sparse(W)$keys

    # Every variable is a vertex, including those without any edges
    n = ncol(W)

    return(.count_clusters(W_keys, n))
}
