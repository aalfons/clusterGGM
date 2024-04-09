#' Estimate Clustered Gaussian Graphical Model without Penalty
#'
#' This function estimates the parameters of a clustered precision matrix based
#' on a restricted negative log-likelihood loss function. The restriction is the
#' clustering provided by the input. This function is different from
#' \code{\link{cggm}}, as there is no penalization of the differences between
#' the different rows of Theta.
#'
#' @param cggm_output Output of \code{\link{cggm}}.
#' @param verbose Determines the amount of information printed during the
#' optimization. Defaults to \code{0}.
#'
#' @return A list containing the estimated parameters of the clustered,
#' unpenalized CGGM model.
#'
#' @examples
#' # Example usage:
#'
#' @seealso \code{\link{cggm}}
#'
#' @export
cggm_refit <- function(cggm_output, verbose = 0)
{
    # Test if input is already result of refitting
    if (class(cggm_output) == "CGGM_refit") {
        return(cggm_output)
    }

    # Indices for unique cluster counts
    indices = match(
        unique(cggm_output$cluster_counts), cggm_output$cluster_counts
    )

    # Prepare result
    refit_result = list()
    refit_result$lambdas = cggm_output$lambdas[indices]
    refit_result$cluster_counts = cggm_output$cluster_counts[indices]
    refit_result$Theta = list()
    refit_result$R = list()
    refit_result$A = list()
    refit_result$clusters = list()

    for (i in 1:length(indices)) {
        ii = indices[i]

        # Prepare input
        R = as.matrix(cggm_output$R[[ii]])
        A = cggm_output$A[[ii]]
        u = cggm_output$clusters[ii, ]
        p = as.numeric(table(u))
        u = u - 1
        W = matrix(0, nrow = nrow(R), ncol = ncol(R))
        W = CGGMR:::.convert_to_sparse(W)

        # Execute algorithm
        result = CGGMR:::.cggm(
            W_keys = W$keys, W_values = W$values, Ri = R, Ai = A, pi = p,
            ui = u, S = cggm_output$inputs$S, lambdas = c(0), eps_fusions = 0,
            scale_factor = 0, gss_tol = cggm_output$inputs$gss_tol,
            conv_tol = cggm_output$inputs$conv_tol,
            max_iter = cggm_output$inputs$max_iter, store_all_res = TRUE,
            verbose = verbose
        )

        # Convert result
        result = CGGMR:::.convert_cggm_output(result)

        # Add to the main result
        refit_result$Theta[[i]] = result$Theta[[1]]
        refit_result$R[[i]] = result$R[[1]]
        refit_result$A[[i]] = result$A[[1]]
        refit_result$clusters[[i]] = result$clusters
    }

    # Convert the list of cluster IDs to a matrix
    refit_result$clusters = do.call(rbind, refit_result$clusters)

    # Create a vector where the nth element contains the index of the solution
    # where n clusters are found for the first time. If an element is -1, that
    # number of clusters is not found
    cluster_solution_index = rep(-1, nrow(cggm_output$inputs$S))
    for (i in 1:length(refit_result$cluster_counts)) {
        c = refit_result$cluster_counts[i]

        if (cluster_solution_index[c] < 0) {
            cluster_solution_index[c] = i
        }
    }
    refit_result$cluster_solution_index = cluster_solution_index

    # The number of solutions
    refit_result$n = length(refit_result$cluster_counts)

    # Rename row and colnames of Theta
    for (i in 1:refit_result$n) {
        rownames(refit_result$Theta[[i]]) = rownames(cggm_output$inputs$S)
        colnames(refit_result$Theta[[i]]) = colnames(cggm_output$inputs$S)
    }

    # Rename the columns of the cluster ID matrix
    colnames(refit_result$clusters) = colnames(cggm_output$inputs$S)

    # Set the class
    class(refit_result) = "CGGM_refit"

    return(refit_result)
}
