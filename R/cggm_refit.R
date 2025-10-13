#' Refit the Gaussian Graphical Model for a Given Aggregation and Sparsity Structure
#'
#' Estimate the parameters of a clustered and sparse precision matrix or
#' covariance matrix based on a restricted negative log-likelihood loss
#' function. The restrictions are given by the provided aggregation and
#' sparsity structure. This function is different from \code{\link{cggm}()},
#' as there are no aggregation and sparsity penalties on the precision or
#' covariance matrix.
#'
#' @param cggm_output An object of class \code{"CGGM"} as returned by
#' \code{\link{cggm}()}.
#' @param verbose Determines the amount of information printed during the
#' optimization. Defaults to \code{0}.
#'
#' @return An object of class \code{"CGGM_refit"} with the following components:
#' \item{A,R}{Lists of matrices. Each pair of matrices with the same index
#' parametrize the estimated precision matrix after the refitting step given
#' the aggregation structure found with the corresponding value of the
#' aggregation parameter \code{lambda_cpath} (and sparsity structure found with
#' the value of the sparsity parameter \code{lambda_lasso}). It is not
#' recommended to use these directly, instead use the accessor function
#' \code{\link{get_Theta}()} to extract the estimated precision matrix for a
#' given index of the aggregation parameter.}
#' \item{clusters}{An integer matrix in which each row contains the cluster
#' assignment of each variable for the corresponding value of the aggregation
#' parameter \code{lambda_cpath}. Use the accessor function
#' \code{\link{get_clusters}()} to extract the cluster assignment for a given
#' index of the aggregation parameter.}
#' \item{lambdas}{A vector with the values for the aggregation parameter
#' \code{lambda_cpath} for which the CGGM loss function has been  minimized.}
#' \item{Theta}{List of matrices. Contains the solution to the minimization
#' procedure for each value of the aggregation parameter \code{lambda_cpath}.
#' It is not recommended to use these directly, instead use the accessor
#' function \code{\link{get_Theta}()} to extract the estimated precision matrix
#' for a given index of the aggregation parameter.}
#' \item{cluster_counts}{An integer vector containing the number of clusters
#' obtained for each value of the aggregation parameter \code{lambda_cpath}.}
#' \item{cluster_solution_index}{An integer vector containing the index of the
#' value of the aggregation parameter \code{lambda_cpath} for which a certain
#' number of clusters was attained. For example,
#' \code{cluster_solution_index[2]} yields the index of the smallest value for
#' \code{lambda_cpath} for which a solution with two clusters was found.
#' Contains -1 if there is no value for  \code{lambda_cpath} with that number
#' of clusters.}
#' \item{n}{The number of values of the aggregation parameter
#' \code{lambda_cpath} for which the CGGM loss function was minimized.}
#'
#' @note The function interface and output structure are still experimental and
#' may change in the next version.
#'
#' @author Daniel J.W. Touw
#'
#' @references
#' D.J.W. Touw, A. Alfons, P.J.F. Groenen and I. Wilms (2025)
#' \emph{Clusterpath Gaussian Graphical Modeling}. arXiv:2407.00644.
#' doi:10.48550/arXiv.2407.00644.
#'
#' @seealso
#' \code{\link{cggm}()}, \code{\link{cggm_cv}()}
#'
#' @example inst/doc/examples/example-cggm_refit.R
#'
#' @useDynLib clusterGGM
#' @export
cggm_refit <- function(cggm_output, verbose = 0)
{
    # Test if input is already result of refitting
    if (inherits(cggm_output, "CGGM_refit")) {
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
        W_cpath = matrix(0, nrow = nrow(R), ncol = ncol(R))
        W_cpath = .convert_to_sparse(W_cpath)
        W_lasso = matrix(0, nrow = nrow(R), ncol = ncol(R))
        refit_lasso = matrix(1, nrow = nrow(R), ncol = ncol(R))

        # If the lasso penalty was active, make sure R has a sparsity structure
        # and set the flags in the matrix that records which elements of R
        # should be updated to zero
        if (cggm_output$inputs$lambda_lasso > 0) {
            R[abs(R) < cggm_output$inputs$eps_lasso] = 0
            refit_lasso[abs(R) < cggm_output$inputs$eps_lasso] = 0
        }

        # Execute algorithm
        result = .cggm(
            W_keys = W_cpath$keys, W_values = W_cpath$values,
            W_lassoi = W_lasso, Ri = R, Ai = A, pi = p, ui = u,
            S = cggm_output$inputs$S, lambdas = c(0), lambda_lasso = 0,
            eps_lasso = cggm_output$inputs$eps_lasso, eps_fusions = 0,
            scale_factor_cpath = 0, scale_factor_lasso = 0,
            gss_tol = cggm_output$inputs$gss_tol,
            conv_tol = cggm_output$inputs$conv_tol / 10,
            max_iter = cggm_output$inputs$max_iter, refit = TRUE,
            refit_lasso = refit_lasso, store_all_res = TRUE, verbose = verbose
        )

        # Convert result
        result = .convert_cggm_output(result)

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
