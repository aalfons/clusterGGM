#' Estimate Clusterpath Gaussian Graphical Model
#'
#' This function estimates the parameters of a CGGM model based on the input
#' data and parameters.
#'
#' @param S The sample covariance matrix of the data.
#' @param W_cpath The weight matrix used in the clusterpath penalty.
#' @param lambda_cpath A numeric vector of tuning parameters for regularization.
#' Should be a sequence of monotonically increasing values.
#' @param W_lasso The weight matrix used in the lasso penalty. Defaults to
#' \code{NULL}, which is interpreted as all weights being zero (no
#' penalization).
#' @param lambda_lasso The penalty parameter used for the lasso penalty.
#' Defaults to \code{0} (no penalization).
#' @param eps_lasso Parameter that governs the quadratic approximation of the
#' lasso penalty. Within the interval \code{c(-eps_lasso, eps_lasso)} the
#' absolute value function is approximated by a quadratic function. Defaults to
#' \code{5e-3}.
#' @param gss_tol The tolerance value used in the Golden Section Search (GSS)
#' algorithm. Defaults to \code{1e-6}.
#' @param conv_tol The tolerance used to determine convergence. Defaults to
#' \code{1e-7}.
#' @param fusion_threshold The threshold for fusing two clusters. If
#' \code{NULL}, defaults to \code{tau} times the median distance between the
#' rows of \code{solve(S)}.
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
#' solutions of Theta if \code{expand = TRUE}. The difference is computed as
#' \code{norm(Theta[i-1]-Theta[i], "F") / norm(Theta[i-1], "F")}. Defaults to
#' \code{0.01}.
#' @param verbose Determines the amount of information printed during the
#' optimization. Slows down the algorithm significantly. Defaults to \code{0}.
#'
#' @return A list containing the estimated parameters of the CGGM model.
#'
#' @examples
#' # Example usage:
#'
#' @seealso \code{\link{clusterpath_weights}}, \code{\link{lasso_weights}}
#'
#' @export
cggm <- function(S, W_cpath, lambda_cpath, W_lasso = NULL, lambda_lasso = 0,
                 eps_lasso = 5e-3, gss_tol = 5e-3, conv_tol = 1e-7,
                 fusion_threshold = NULL, tau = 1e-3, max_iter = 5000,
                 expand = FALSE, max_difference = 0.01, verbose = 0)
{
    # Check if W_lasso is NULL
    if (is.null(W_lasso)) {
        W_lasso = matrix(0, nrow = nrow(S), ncol = ncol(S))

        # Failsafe for if W_lasso is a zero matrix, prevents a lot of useless
        # computations
        lambda_lasso = 0
    }

    # Compute the CGGM result
    result = .cggm_wrapper(
        S = S, W_cpath = W_cpath, W_lasso = W_lasso,
        lambda_cpath = lambda_cpath, lambda_lasso = lambda_lasso,
        eps_lasso = eps_lasso, gss_tol = gss_tol, conv_tol = conv_tol,
        fusion_threshold = fusion_threshold, tau = tau, max_iter = max_iter,
        store_all_res = TRUE, verbose = verbose
    )

    # If expansion of the solution path is not required, return the current
    # result
    if (!expand) {
        # Rename row and colnames of Theta
        for (i in 1:result$n) {
            rownames(result$Theta[[i]]) = rownames(S)
            colnames(result$Theta[[i]]) = colnames(S)
        }

        # Rename the columns of the cluster ID matrix
        colnames(result$clusters) = colnames(S)

        # Set the class
        class(result) = "CGGM"

        return(result)
    }

    # For now, remove the option to inspect loss progression for expanded
    # results
    result$loss_progression = NULL

    # If expanding the solution path is required, begin with computing the
    # minimum number of clusters attainable given the weight matrix. The first
    # step is to find a value for lambda for which this number is attained.
    target = min_clusters(W_cpath)

    # While the target number of clusters has not been found, continue adding
    # more solutions for larger values of lambda
    while (min(result$cluster_counts) != target && max(result$lambdas) < 1e12) {
        # Get the maximum value of lambda
        max_lambda = max(result$lambdas)
        if (max_lambda == 0) max_lambda = 1

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
        result = .cggm_expand(cggm_output = result, lambda_cpath = lambdas,
                              verbose = 0)
    }

    # Now increase the granularity of lambda. To do so, compute the difference
    # between the consecutive solutions for Theta to determine where additional
    # values for lambda are required.
    diff_norms = rep(0, result$n - 1)
    p = nrow(S)

    for (i in 2:result$n) {
        diff_norms[i - 1] =
            norm(result$Theta[[i - 1]] - result$Theta[[i]], "F") /
            norm(result$Theta[[i - 1]], "F")
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

            # Check if the gap between the lambdas is sufficiently large
            if (min_lambda > 0) {
                if (max_lambda / min_lambda - 1 < 0.01) next
            } else {
                if (max_lambda < 1e-3) next
            }

            # Get the number of lambdas that should be inserted
            n_lambdas = floor(diff_norms[i] / max_difference)

            # Get a sequence that includes the minimum and maximum, and trim
            # those
            lambdas_insert =
                seq(min_lambda, max_lambda, length.out = n_lambdas + 2)
            lambdas = c(lambdas, lambdas_insert[-c(1, n_lambdas + 2)])
        }

        # If there are no new values for lambda, terminate the while loop
        if (length(lambdas) < 1) break

        # Compute additional results
        result = .cggm_expand(cggm_output = result, lambda_cpath = lambdas,
                              verbose = 0)

        # Recompute the differences between the consecutive solutions
        diff_norms = rep(0, result$n - 1)

        for (i in 2:result$n) {
            diff_norms[i - 1] =
                norm(result$Theta[[i - 1]] - result$Theta[[i]], "F") / p
        }
    }

    # Rename row and colnames of Theta
    for (i in 1:result$n) {
        rownames(result$Theta[[i]]) = rownames(S)
        colnames(result$Theta[[i]]) = colnames(S)
    }

    # Rename the columns of the cluster ID matrix
    colnames(result$clusters) = colnames(S)

    # Set the class
    class(result) = "CGGM"

    return(result)
}
