# #' Refit the Gaussian Graphical Model for a Given Aggregation and Sparsity Structure
# #'
# #' Estimate the parameters of a clustered and sparse precision matrix or
# #' covariance matrix based on a restricted negative log-likelihood loss
# #' function. The restrictions are given by the provided aggregation and
# #' sparsity structure. This function is different from \code{\link{cggm}()},
# #' as there are no aggregation and sparsity penalties on the precision or
# #' covariance matrix.
# #'
# #' Each solution in the supplied object is refitted, so that the result can be
# #' indexed in exactly the same way. The refitted solution is fully determined
# #' by the sample covariance matrix, the cluster assignment and the sparsity
# #' structure. Consecutive solutions that share the latter two therefore share a
# #' single minimization of the restricted loss function.
# #'
# #' @param cggm_output An object of class \code{"CGGM"} as returned by
# #' \code{\link{cggm}()}.
# #' @param verbose Determines the amount of information printed during the
# #' optimization. Defaults to \code{0}.
# #'
# #' @return An object of class \code{"CGGM_refit"} with the following components:
# #' \item{A,R}{Lists of matrices. Each pair of matrices with the same index
# #' parametrize the estimated matrix after the refitting step, given
# #' the aggregation and sparsity structure of the solution with the same index
# #' in \code{cggm_output}. It is not recommended to use these directly, instead
# #' use the accessor functions \code{\link{get_precision}()} and
# #' \code{\link{get_covariance}()} to extract the estimated matrix for a given
# #' index of the aggregation parameter.}
# #' \item{clusters}{An integer matrix in which each row contains the cluster
# #' assignment of each variable for the corresponding value of the aggregation
# #' parameter \code{lambda_cpath}. Use the accessor function
# #' \code{\link{get_clusters}()} to extract the cluster assignment for a given
# #' index of the aggregation parameter.}
# #' \item{lambdas}{The values of the aggregation parameter \code{lambda_cpath}
# #' of \code{cggm_output}. The solution with a given index is the refit of the
# #' solution with the same index in \code{cggm_output}. The refitting step
# #' itself does not use an aggregation or a sparsity penalty.}
# #' \item{cluster_counts}{An integer vector containing the number of clusters
# #' obtained for each value of the aggregation parameter \code{lambda_cpath}.}
# #' \item{cluster_solution_index}{An integer vector containing the index of the
# #' value of the aggregation parameter \code{lambda_cpath} for which a certain
# #' number of clusters was attained. For example,
# #' \code{cluster_solution_index[2]} yields the index of the smallest value for
# #' \code{lambda_cpath} for which a solution with two clusters was found.
# #' Contains -1 if there is no value for  \code{lambda_cpath} with that number
# #' of clusters.}
# #' \item{n}{The number of solutions, which equals the number of solutions in
# #' \code{cggm_output}.}
# #' \item{matrix}{A character string indicating which matrix was estimated,
# #' carried over from \code{cggm_output}.}
# #' \item{inputs}{A list of the inputs of the refitting step, with the same
# #' components as the \code{inputs} of \code{\link{cggm}()}. The penalty
# #' parameters and weight matrices are zero, as the refitting step does not
# #' penalize.}
# #'
# #' @note The function interface and output structure are still experimental and
# #' may change in the next version.
# #'
# #' @author Daniel J.W. Touw
# #'
# #' @references
# #' D.J.W. Touw, A. Alfons, P.J.F. Groenen and I. Wilms (2026) Clusterpath
# #' Gaussian Graphical Modeling. \emph{Journal of Computational and Graphical
# #' Statistics}. doi:10.1080/10618600.2026.2653764.
# #'
# #' @seealso
# #' \code{\link{cggm}()}, \code{\link{cggm_cv}()}
# #'
# #' @example inst/doc/examples/example-cggm_refit.R
# #'
# #' @useDynLib clusterGGM
# #' @export
# cggm_refit <- function(cggm_output, verbose = 0)
# {
#     # Test if input is already result of refitting
#     if (inherits(cggm_output, "CGGM_refit")) {
#         warning("Input is already a refitted result, it is returned unchanged")
#         return(cggm_output)
#     }
#
#     n = cggm_output$n
#     lambda_lasso = cggm_output$inputs$lambda_lasso
#     eps_lasso = cggm_output$inputs$eps_lasso
#
#     # Magnitude of the starting value of the refits, so that it does not depend
#     # on the units of the data
#     start_scale = mean(diag(.initial_Theta(cggm_output$inputs$S)))
#
#     # Prepare result
#     refit_result = list()
#     refit_result$lambdas = cggm_output$lambdas
#     refit_result$R = vector("list", n)
#     refit_result$A = vector("list", n)
#     clusters = vector("list", n)
#
#     # Structure of the solution that was refitted most recently
#     prev_u = NULL
#     prev_mask = NULL
#
#     for (i in 1:n) {
#         # Prepare input
#         R = as.matrix(cggm_output$R[[i]])
#         A = cggm_output$A[[i]]
#         u = cggm_output$clusters[i, ]
#
#         # Elements of R that the sparsity penalty shrunk to zero
#         if (lambda_lasso > 0) {
#             mask = abs(R) < eps_lasso
#         } else {
#             mask = matrix(FALSE, nrow = nrow(R), ncol = ncol(R))
#         }
#
#         # An unchanged structure yields the same refitted solution
#         if (identical(u, prev_u) && identical(mask, prev_mask)) {
#             refit_result$R[[i]] = refit_result$R[[i - 1]]
#             refit_result$A[[i]] = refit_result$A[[i - 1]]
#             clusters[[i]] = clusters[[i - 1]]
#             next
#         }
#
#         p = as.numeric(table(u))
#         W_cpath = matrix(0, nrow = nrow(R), ncol = ncol(R))
#         W_cpath = .convert_to_sparse(W_cpath)
#         W_lasso = matrix(0, nrow = nrow(R), ncol = ncol(R))
#
#         # Start every refit from a scaled identity matrix, so that a given
#         # aggregation and sparsity structure always yields the same solution
#         A = rep(start_scale, max(u))
#         R = matrix(0, nrow = max(u), ncol = max(u))
#
#         # Impose the sparsity structure on R and set the flags in the matrix
#         # that records which elements of R should be updated to zero
#         R[mask] = 0
#         refit_lasso = matrix(1, nrow = nrow(R), ncol = ncol(R))
#         refit_lasso[mask] = 0
#
#         # Execute algorithm
#         result = .cggm(
#             W_keys = W_cpath$keys, W_values = W_cpath$values,
#             W_lassoi = W_lasso, Ri = R, Ai = A, pi = p, ui = u - 1,
#             S = cggm_output$inputs$S, lambdas = c(0), lambda_lasso = 0,
#             eps_lasso = cggm_output$inputs$eps_lasso, eps_fusions = 0,
#             scale_factor_cpath = 0, scale_factor_lasso = 0,
#             gss_tol = cggm_output$inputs$gss_tol,
#             conv_tol = cggm_output$inputs$conv_tol / 10,
#             max_iter = cggm_output$inputs$max_iter, refit = TRUE,
#             refit_lasso = refit_lasso, store_all_res = TRUE, verbose = verbose
#         )
#
#         # Convert result
#         result = .convert_cggm_output(result)
#
#         # Add to the main result
#         refit_result$R[[i]] = result$R[[1]]
#         refit_result$A[[i]] = result$A[[1]]
#         clusters[[i]] = result$clusters
#
#         prev_u = u
#         prev_mask = mask
#     }
#
#     # Convert the list of cluster IDs to a matrix
#     refit_result$clusters = do.call(rbind, clusters)
#
#     # The number of clusters in each of the refitted solutions
#     refit_result$cluster_counts = apply(
#         refit_result$clusters, 1, function(x) length(unique(x))
#     )
#
#     # Create a vector where the nth element contains the index of the solution
#     # where n clusters are found for the first time. If an element is -1, that
#     # number of clusters is not found
#     cluster_solution_index = rep(-1, nrow(cggm_output$inputs$S))
#     for (i in 1:length(refit_result$cluster_counts)) {
#         c = refit_result$cluster_counts[i]
#
#         if (cluster_solution_index[c] < 0) {
#             cluster_solution_index[c] = i
#         }
#     }
#     refit_result$cluster_solution_index = cluster_solution_index
#
#     # The number of solutions
#     refit_result$n = length(refit_result$cluster_counts)
#
#     # The refitting step estimates the same matrix as the supplied result
#     refit_result$matrix = cggm_output$matrix
#
#     # The inputs of the refitting step, which is not penalized
#     p_variables = nrow(cggm_output$inputs$S)
#     refit_result$inputs = list(
#         S = cggm_output$inputs$S,
#         W_cpath = matrix(0, nrow = p_variables, ncol = p_variables),
#         gss_tol = cggm_output$inputs$gss_tol,
#         conv_tol = cggm_output$inputs$conv_tol / 10,
#         max_iter = cggm_output$inputs$max_iter,
#         lambda_lasso = 0,
#         eps_lasso = cggm_output$inputs$eps_lasso,
#         W_lasso = matrix(0, nrow = p_variables, ncol = p_variables)
#     )
#
#     # Rename the columns of the cluster ID matrix
#     colnames(refit_result$clusters) = colnames(cggm_output$inputs$S)
#
#     # Set the class
#     class(refit_result) = c("CGGM_refit", "CGGM_path")
#
#     return(refit_result)
# }
