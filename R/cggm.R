#' Estimate Clusterpath Gaussian Graphical Model
#'
#' This function estimates the parameters of a CGGM model based on the input
#' data and parameters.
#'
#' @param S The sample covariance matrix of the data.
#' @param W The weight matrix used in the clusterpath penalty.
#' @param lambda A numeric vector of tuning parameters for regularization.
#' Make sure the values are monotonically increasing.
#' @param gss_tol The tolerance value used in the Golden Section Search (GSS)
#' algorithm. Defaults to \code{1e-6}.
#' @param conv_tol The tolerance used to determine convergence. Defaults to
#' \code{1e-7}.
#' @param fusion_threshold The threshold for fusing two clusters. If NULL,
#' defaults to \code{tau} times the median distance between the rows of
#' \code{solve(S)}.
#' @param tau The parameter used to determine the fusion threshold. Defaults to
#' \code{1e-3}.
#' @param max_iter The maximum number of iterations allowed for the optimization
#' algorithm. Defaults to \code{5000}.
#' @param expand Determines whether the vector \code{lambda} should be expanded
#' with additional values in order to find a sequence of solutions that (a)
#' terminates in the minimum number of clusters and (b) has consecutive
#' solutions for Theta that are not too different from each other. The degree
#' of difference between consecutive solutions that is allowed is determined by
#' \code{max_difference}. Defaults to \code{FALSE}.
#' @param max_difference The maximum allowed difference between consecutive
#' solutions of Theta. The difference is computed as
#' \code{norm(Theta[i]-Theta[i], "F") / nrow(Theta)}. Defaults to \code{0.01}.
#' @param verbose Determines the amount of information printed during the
#' optimization. Slows down the algorithm significantly. Defaults to \code{0}.
#'
#' @return A list containing the estimated parameters of the CGGM model.
#'
#' @examples
#' # Example usage:
#'
#' @seealso \code{\link{cggm_weights}}
#'
#' @export
cggm <- function(S, W, lambda, gss_tol = 1e-6, conv_tol = 1e-7,
                 fusion_threshold = NULL, tau = 1e-3, max_iter = 5000,
                 expand = FALSE, max_difference = 0.01, verbose = 0)
{
    # Compute the CGGM result
    result = CGGMR:::.cggm_wrapper(
        S = S, W = W, lambda = lambda, gss_tol = gss_tol, conv_tol = conv_tol,
        fusion_threshold = fusion_threshold, tau = tau, max_iter,
        store_all_res = TRUE, verbose = verbose
    )

    # If expansion of the solution path is not required, return the current
    # result
    if (!expand) {
        # Set the class
        class(result) = "CGGM"

        return(result)
    }

    # If expanding the solution path is required, begin with computing the
    # minimum number of clusters attainable given the weight matrix. The first
    # step is to find a value for lambda for which this number is attained.
    target = min_clusters(W)

    # While the target number of clusters has not been found, continue adding
    # more solutions for larger values of lambda
    while (min(result$cluster_counts) != target) {
        # Get the maximum value of lambda
        max_lambda = max(result$lambdas)
        if (max_lambda == 0) max_lambda = 1

        # Set an additional sequence of values, do this by linear interpolation
        # lambdas = seq(max_lambda, 2 * max_lambda, length.out = 5)[-1]

        # Increase maximum lambda to factor_max times the previous largest value
        # and do this in steps of at most factor_step times the previous value
        # for lambda
        factor_max = 1.5
        factor_step = 1.05
        n_steps = ceiling(log(factor_max) / log(factor_step))
        factor_step_mod = exp(log(factor_max) / n_steps)

        # Set an additional sequence of values
        lambdas = max_lambda * factor_step_mod^seq(n_steps)

        # Compute additional results
        result = CGGMR:::.cggm_expand(cggm_output = result, lambdas = lambdas,
                                      verbose = 0)
    }

    # Now increase the granularity of lambda. To do so, compute the difference
    # between the consecutive solutions for Theta to determine where additional
    # values for lambda are required.
    diff_norms = rep(0, result$n - 1)
    p = nrow(S)

    for (i in 2:result$n) {
        diff_norms[i - 1] =
            norm(result$Theta[[i - 1]] - result$Theta[[i]], "F") / p
    }

    # Repeat adding solutions until none of the differences exceed the maximum
    # allowed difference.
    while (any(diff_norms > max_difference)) {
        # Find the differences that exceed the maximum
        indices = which(diff_norms > max_difference)

        # Select lambdas to fill in the gaps. The number of inserted lambdas
        # depends linearly on the magnitude of the difference between two
        # consecutive solutions.
        lambdas = c()

        for (i in indices) {
            # Get minimum and maximum value of lambda
            min_lambda = result$lambdas[i]
            max_lambda = result$lambdas[i + 1]

            # Get the number of lambdas that should be inserted
            n_lambdas = floor(diff_norms[i] / max_difference)

            # Get a sequence that includes the minimum and maximum, and trim
            # those
            lambdas_insert =
                seq(min_lambda, max_lambda, length.out = n_lambdas + 2)
            lambdas = c(lambdas, lambdas_insert[-c(1, n_lambdas + 2)])
        }

        # Compute additional results
        result = CGGMR:::.cggm_expand(cggm_output = result, lambdas = lambdas,
                                      verbose = 0)

        # Recompute the differences between the consecutive solutions
        diff_norms = rep(0, result$n - 1)

        for (i in 2:result$n) {
            diff_norms[i - 1] =
                norm(result$Theta[[i - 1]] - result$Theta[[i]], "F") / p
        }
    }

    # Set the class
    class(result) = "CGGM"

    return(result)
}
