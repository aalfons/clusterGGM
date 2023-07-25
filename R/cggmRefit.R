#' Estimate Clusterpath Gaussian Graphical Model
#'
#' This function estimates the parameters of a CGGM model based on the input
#' data and parameters.
#'
#' @param cggm_output Output of cggm().
#' @param S The sample covariance matrix of the data.
#' @param gss_tol The tolerance value used in the Golden Section Search (GSS)
#' algorithm. Defaults to \code{1e-4}.
#' @param conv_tol The tolerance used to determine convergence. Defaults to
#' \code{1e-9}.
#' @param max_iter The maximum number of iterations allowed for the optimization
#' algorithm. Defaults to \code{2000}.
#' @param use_Newton Logical, indicating whether to use Newton's method in the
#' optimization algorithm. Defaults to \code{TRUE}.
#' @param verbose Determines the amount of information printed during the
#' optimization. Slows down the algorithm significantly. Defaults to \code{0}.
#'
#' @return A list containing the estimated parameters of the CGGM model.
#'
#' @examples
#' # Example usage:
#'
#' @export
cggmRefit <- function(cggm_output, S, gss_tol = 1e-4, conv_tol = 1e-9,
                      max_iter = 2000, use_Newton = TRUE, verbose = 0)
{
    # Indices for unique cluster counts
    indices = match(
        unique(cggm_output$cluster_counts), cggm_output$cluster_counts
    )

    # Prepare result
    refit_result = list()
    refit_result$cluster_counts = cggm_output$cluster_counts[indices]
    refit_result$Theta = list()
    refit_result$R = list()
    refit_result$A = list()
    refit_result$clusters = list()

    for (i in 1:length(indices)) {
        ii = indices[i]

        # Prepare input
        R = cggm_output$R[[ii]]
        A = cggm_output$A[[ii]]
        u = cggm_output$clusters[[ii]]
        p = as.numeric(table(u))
        u = u - 1
        W = matrix(0, nrow = nrow(R), ncol = ncol(R))

        # Execute algorithm
        result = CGGMR:::.cggm(
            Ri = R, Ai = A, pi = p, ui = u, S = S, UWUi = W, lambdas = c(0),
            gss_tol = gss_tol, conv_tol = conv_tol,
            fusion_check_threshold = 0, max_iter = max_iter,
            store_all_res = TRUE, verbose = verbose,
            print_profile_report = FALSE, fusion_type = 3,
            Newton_dd = use_Newton
        )

        # Convert result
        result = CGGMR:::.convertCGGMOutput(result)

        # Add to the main result
        refit_result$Theta[[i]] = result$Theta[[1]]
        refit_result$R[[i]] = result$R[[1]]
        refit_result$A[[i]] = result$A[[1]]
        refit_result$clusters = result$clusters[[1]]
    }

    # Create a vector where the nth element contains the index of the solution
    # where n clusters are found for the first time. If an element is -1, that
    # number of clusters is not found
    cluster_solution_index = rep(-1, nrow(S))
    for (i in 1:length(refit_result$cluster_counts)) {
        c = refit_result$cluster_counts[i]

        if (cluster_solution_index[c] < 0) {
            cluster_solution_index[c] = i
        }
    }
    refit_result$cluster_solution_index = cluster_solution_index

    # The number of solutions
    refit_result$n = length(refit_result$cluster_counts)

    return(refit_result)
}
