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
#' \code{1e-9}.
#' @param fusion_type The type of fusion to be used for determining fusion
#' between two objects. Possible values: "proximity" (based on closeness) or
#' one of {"a0", "a1", "a2"} (based on analytical evaluation). Defaults to
#' "proximity".
#' @param fusion_threshold If \code{fusion_type = "proximity"}, it is the
#' threshold for fusing two clusters. For the analytical fusions, it is the
#' threshold below which the analytical check is executed. If NULL, defaults
#' to \code{sqrt(nrow(S)) * 1e-4} times the median distance between the rows of
#' \code{solve(S)} for proximity based fusions and to \code{sqrt(nrow(S))} times
#' the median distance for analytical fusions.
#' @param max_iter The maximum number of iterations allowed for the optimization
#' algorithm. Defaults to \code{2000}.
#' @param store_all_res Logical, indicating whether to store the results for all
#' values for lambda. If false, only solutions for different numbers of
#' clusters are stored. Defaults to \code{FALSE}.
#' @param use_Newton Logical, indicating whether to use Newton's method in the
#' optimization algorithm. Defaults to \code{TRUE}.
#' @param profile Logical, indicates whether a profiling report should be
#' printed. Defaults to \code{FALSE}.
#' @param verbose Determines the amount of information printed during the
#' optimization. Slows down the algorithm significantly. Defaults to \code{0}.
#'
#' @return A list containing the estimated parameters of the CGGM model.
#'
#' @examples
#' # Example usage:
#'
#' @export
cggm <- function(S, W, lambdas, gss_tol = 1e-4, conv_tol = 1e-9,
                 fusion_type = "proximity", fusion_threshold = NULL,
                 max_iter = 2000, store_all_res = FALSE, use_Newton = TRUE,
                 profile = FALSE, verbose = 0)
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

        if (fusion_type == "proximity") {
            # Set the fusion_threshold to a small value relative to the median
            # distance as threshold for fusions, if the median is too small,
            # i.e., when Theta is mostly clustered into a single cluster, a
            # buffer is added
            fusion_threshold = 1e-4 * max(m, 1e-8) * sqrt(nrow(S))
        } else {
            # Set fusion_threshold to m, if analytical fusions are used, this is
            # the threshold to execute the analytical check
            fusion_threshold = max(m, 0.1) * sqrt(nrow(S))
        }
    }

    # Set the fusion_type argument for the C++ function
    if (fusion_type == "proximity") {
        fusion_type_int = 3
    } else if (fusion_type == "a0") {
        fusion_type_int = 0
    } else if (fusion_type == "a1") {
        fusion_type_int = 1
    } else if (fusion_type == "a2") {
        fusion_type_int = 2
    } else {
        message = "fusion_type should one of 'proximity', 'a0', 'a1', 'a2'"
        stop(message)
    }

    # Execute algorithm
    result = CGGMR:::.cggm(
        Ri = R, Ai = A, pi = p, ui = u, S = S, UWUi = W, lambdas = lambdas,
        gss_tol = gss_tol, conv_tol = conv_tol,
        fusion_check_threshold = fusion_threshold, max_iter = max_iter,
        store_all_res = store_all_res, verbose = verbose,
        print_profile_report = profile, fusion_type = fusion_type_int,
        Newton_dd = use_Newton
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
    if (fusion_type == "proximity") {
        result$fusion_threshold = fusion_threshold * sqrt(nrow(S))
    }

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
