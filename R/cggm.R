#' Estimate Clusterpath Gaussian Graphical Model
#'
#' This function estimates the parameters of a CGGM model based on the input
#' data and parameters.
#'
#' @param S The sample covariance matrix of the data.
#' @param W The weight matrix used in the clusterpath penalty.
#' @param lambdas A numeric vector of tuning parameters for regularization.
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
#' to 1e-4 * the median distance between the rows of \code{solve(S)} for
#' proximity based fusions and to the median distance for analytical
#' based fusions.
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
#' cggm_result <- cggm(S, W, lambdas)
#'
#' @export
cggm <- function(S, W, lambdas, gss_tol = 1e-4, conv_tol = 1e-9,
                 fusion_type = "proximity", fusion_threshold = NULL,
                 max_iter = 2000, store_all_res = FALSE, use_Newton = TRUE,
                 profile = FALSE, verbose = 0)
{
    # Initial estimate for Theta
    Theta = tryCatch(
        {
            # Try to compute the inverse using solve(X)
            inv_S = solve(S)
        },
        error = function(e) {
            # In case of an error (non-invertible matrix), use solve(S + I)
            inv_S = solve(S + diag(nrow(S)))

            # Print warning
            warning(
                "In cggm: S is singular, Theta is initialized as (S + I)^-1",
                call. = FALSE
            )
        }
    )

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
        m = median(sqrt(-log(weightsTheta(Theta, 1)[lower.tri(Theta)])))

        # Set fusion_threshold to m, if analytical fusions are used, this is the
        # threshold to execute the analytical check
        fusion_threshold = m

        # Set the fusion_threshold to a much smaller value if proximity is used
        # to fuse
        if (fusion_type == "proximity") {
            fusion_threshold = 1e-4 * m
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
    result = CGGMR::convertCGGMOutput(result)
    result$losses = losses
    result$lambdas = lambdas_res
    result$cluster_counts = cluster_counts

    return(result)
}
