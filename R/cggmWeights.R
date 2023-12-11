#' Compute weight matrix for the Clusterpath Gaussian Graphical Model
#'
#' This function computes the (possibly sparse) weight matrix that is used in
#' CGGM. Weights are computed using exp{-phi * norm(Theta, i, j)^2}, where the
#' squared norm of two rows of Theta are scaled by the average squared norm of
#' all the rows of Theta.
#'
#' @param S The sample covariance matrix of the data.
#' @param phi Tuning parameter of the weights.
#' @param k The number of nearest neighbors that should be used to set weights
#' to a nonzero value. If \code{0 < k < ncol(S)}, the dense weight matrix will
#' be made sparse, otherwise the dense matrix is returned.
#'
#' @return A weight matrix.
#'
#' @examples
#' # Example usage:
#'
#' @export
cggmWeights <- function(S, phi, k)
{
    # Initial estimate for Theta
    Theta = CGGMR:::.initial_Theta(S)

    # Get dense weight matrix, if phi = 0, the knn part breaks so, in that case,
    # start with a phi != 0 and correct later
    if (phi > 0) {
        result = CGGMR:::.weights_Theta(Theta, phi)
    } else {
        result = CGGMR:::.weights_Theta(Theta, 1)
    }

    # If k is a sensible value, sparsify the weight matrix
    if (k > 0 && k < nrow(Theta)) {
        result_sparse = matrix(0, nrow = nrow(result), ncol = ncol(result))

        # Select k largest elements in each row and column
        for (i in 1:nrow(result)) {
            indices = CGGMR:::.k_largest(result[, i], k)
            result_sparse[i, indices] = result[i, indices]
            result_sparse[indices, i] = result[indices, i]
        }

        # Store result in the correct variable
        result = result_sparse
    }

    # If phi = 0, set the nonzero weights in the result to 1
    if (phi == 0) {
        result[result != 0] = 1
    }

    return(result)
}
