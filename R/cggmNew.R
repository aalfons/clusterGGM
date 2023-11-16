#' Estimate Clusterpath Gaussian Graphical Model
#'
#' This function estimates the parameters of a CGGM model based on the input
#' data and parameters.
#'
#' @param S The sample covariance matrix of the data.
#' @param W The weight matrix used in the clusterpath penalty.
#' @param lambdas A numeric vector of tuning parameters for regularization.
#' Make sure the values are monotonically increasing.
#' @param gss_tol The tolerance value used in the Golden Section Search (GSS)
#' algorithm. Defaults to \code{1e-4}.
#' @param conv_tol The tolerance used to determine convergence. Defaults to
#' \code{1e-7}.
#' @param fusion_threshold The threshold for fusing two clusters. If NULL,
#' defaults to \code{tau} times the median distance between the rows of
#' \code{solve(S)}.
#' @param tau The parameter used to determine the fusion threshold. Defaults to
#' \code{1e-3}.
#' @param max_iter The maximum number of iterations allowed for the optimization
#' algorithm. Defaults to \code{5000}.
#' @param store_all_res Logical, indicating whether to store the results for all
#' values for lambda. If false, only solutions for different numbers of
#' clusters are stored. Defaults to \code{FALSE}.
#' @param verbose Determines the amount of information printed during the
#' optimization. Slows down the algorithm significantly. Defaults to \code{0}.
#'
#' @return A list containing the estimated parameters of the CGGM model.
#'
#' @examples
#' # Example usage:
#'
#' @export
cggmNew <- function(S, W, lambdas, gss_tol = 1e-4, conv_tol = 1e-7,
                    fusion_threshold = NULL, tau = 1e-3, max_iter = 5000,
                    store_all_res = FALSE, verbose = 0)
{
    # Initial estimate for Theta
    Theta = CGGMR:::.initialTheta(S)

    # Extract R and A from Theta
    R = Theta
    diag(R) = 0
    A = diag(Theta)

    # Each observation starts as its own cluster
    u = c(1:ncol(S)) - 1

    # All cluster sizes are 1
    p = rep(1, ncol(S))

    if (is.null(fusion_threshold)) {
        # Get the median distance between two rows/columns in Theta
        m = CGGMR:::.medianDistance(Theta)

        # Set the fusion_threshold to a small value relative to the median
        # distance as threshold for fusions, if the median is too small,
        # i.e., when Theta is mostly clustered into a single cluster, a
        # buffer is added
        fusion_threshold = tau * max(m, 1e-8)
    }

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

    # Execute algorithm
    result = CGGMR:::.cggm2(
        W_keys = W_keys, W_values = W_values, Ri = R, Ai = A, pi = p, ui = u,
        S = S, lambdas = lambdas, eps_fusions = fusion_threshold,
        conv_tol = conv_tol, max_iter = max_iter, verbose = verbose
    )

    # Convert output
    losses = result$losses
    lambdas_res = result$lambdas
    cluster_counts = result$cluster_counts
    result = CGGMR:::.convertCGGMOutput(result)
    result$losses = losses
    result$lambdas = lambdas_res
    result$cluster_counts = cluster_counts

    # If proximity based clustering is used, also add the fusion threshold to
    # the result
    result$fusion_threshold = fusion_threshold

    # Create a vector where the nth element contains the index of the solution where
    # n clusters are found for the first time. If an element is -1, that number of
    # clusters is not found
    cluster_solution_index = rep(-1, nrow(S))
    for (i in 1:length(result$cluster_counts)) {
        c = result$cluster_counts[i]

        if (cluster_solution_index[c] < 0) {
            cluster_solution_index[c] = i
        }
    }
    result$cluster_solution_index = cluster_solution_index

    # The number of solutions
    result$n = length(cluster_counts)

    return(result)
}
