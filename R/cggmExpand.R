#' Estimate Clusterpath Gaussian Graphical Model
#'
#' This function estimates expands a previous solutionpath by minimizing the
#' loss function for additional values for lambda. It uses warm starts extracted
#' from a previous result to reduce computational burden.
#'
#' @param cggm_output Output of the cggmNew function.
#' @param lambdas Additional lambdas for which the loss should be minimized.
#' Values for which there already is a solution are discarded as well as those
#' that are smaller than the smallest value: \code{min(cggm_output$lambdas)}.
#' @param verbose Determines the amount of information printed during the
#' optimization. Defaults to \code{0}.
#'
#' @return A list containing the estimated parameters of the CGGM model.
#'
#' @examples
#' # Example usage:
#'
#' @export
cggm_expand <- function(cggm_output, lambdas, verbose = 0)
{
    # Remove lambdas for which there is already a solution
    new_lambdas = lambdas[!(lambdas %in% cggm_output$lambdas)]

    if (length(new_lambdas) == 0) {
        return(cggm_output)
    }

    # Remove lambdas that are smaller than the smallest lambda for which there
    # is already a solution
    lambdas = lambdas[lambdas > min(cggm_output$lambdas)]

    if (length(new_lambdas) == 0) {
        return(cggm_output)
    }

    # Indices for the values for the largest value for lambda that is smaller
    # than the one in the new vector
    indices = rep(0, length(new_lambdas))

    for (i in 1:length(new_lambdas)) {
        # Add the index
        is_larger = new_lambdas[i] >= cggm_output$lambdas

        if (sum(is_larger) > 0) {
            indices[i] = which(is_larger)[sum(is_larger)]
        }
    }

    # Initialize the list of additional results
    new_results = list()

    # Initialize the index for the value of the new_lambdas vector for which the
    # loss should be minimized
    index = 1

    for (j in 1:length(unique(indices))) {
        # In the following bit, find the sequence for lambda that share the same
        # index for the warm start. At the end of the loop, lambdas will contain
        # the values that share warm_start_index as warm start, and index will
        # contain the index of the first value in new_lambdas that needs a new
        # warm start
        lambdas = c(new_lambdas[index])
        warm_start_index = indices[index]

        if (index < length(indices)) {
            for (i in (index + 1):length(indices)) {
                if (indices[i] == indices[i - 1]) {
                    lambdas = c(lambdas, new_lambdas[i])
                } else {
                    index = i
                    break
                }
            }
        }

        # Select warm start variables
        R = cggm_output$R[[warm_start_index]]
        A = cggm_output$A[[warm_start_index]]
        u = cggm_output$clusters[[warm_start_index]] - 1
        p = as.numeric(table(u))

        # Membership matrix
        U = matrix(0, nrow = length(u), ncol = max(u) + 1)
        U[cbind(seq_along(u + 1), u + 1)] = 1

        # Clustered weight matrix
        UWU = t(U) %*% cggm_output$inputs$W %*% U
        diag(UWU) = 0
        UWU = CGGMR:::.convert_to_sparse(UWU)

        # Execute algorithm
        result_extra = CGGMR:::.cggm2(
            W_keys = UWU$keys, W_values = UWU$values, Ri = R, Ai = A, pi = p,
            ui = u, S = cggm_output$inputs$S, lambdas = lambdas,
            eps_fusions = cggm_output$fusion_threshold,
            gss_tol = cggm_output$inputs$gss_tol,
            conv_tol = cggm_output$inputs$conv_tol,
            max_iter = cggm_output$inputs$max_iter, store_all_res = TRUE,
            verbose = verbose
        )

        # Convert output
        losses = result_extra$losses
        lambdas_res = result_extra$lambdas
        cluster_counts = result_extra$cluster_counts
        result_extra = CGGMR:::.convertCGGMOutput(result_extra)
        result_extra$losses = losses
        result_extra$lambdas = lambdas_res
        result_extra$cluster_counts = cluster_counts
        result_extra$warm_start_index = warm_start_index

        # Store extra output in an additional list, results are merged at the
        # end
        new_results[[length(new_results) + 1]] = result_extra
    }

    for (i in length(new_results):1) {
        index = new_results[[i]]$warm_start_index

        # Add solutions to the original result
        if (index == length(cggm_output$losses)) {
            cggm_output$losses = c(cggm_output$losses,
                                   new_results[[i]]$losses)
            cggm_output$lambdas = c(cggm_output$lambdas,
                                    new_results[[i]]$lambdas)
            cggm_output$cluster_counts = c(cggm_output$cluster_counts,
                                           new_results[[i]]$cluster_counts)
        } else {
            end = length(cggm_output$losses)
            cggm_output$losses = c(cggm_output$losses[1:index],
                                   new_results[[i]]$losses,
                                   cggm_output$losses[(index + 1):end])
            cggm_output$lambdas = c(cggm_output$lambdas[1:index],
                                    new_results[[i]]$lambdas,
                                    cggm_output$lambdas[(index + 1):end])
            cggm_output$cluster_counts = c(
                cggm_output$cluster_counts[1:index],
                new_results[[i]]$cluster_counts,
                cggm_output$cluster_counts[(index + 1):end]
            )
        }
        cggm_output$Theta = append(cggm_output$Theta, new_results[[i]]$Theta,
                              after = index)
        cggm_output$R = append(cggm_output$R, new_results[[i]]$R, after = index)
        cggm_output$A = append(cggm_output$A, new_results[[i]]$A, after = index)
        cggm_output$clusters = append(cggm_output$clusters,
                                      new_results[[i]]$clusters,
                                      after = index)
    }

    # Create a vector where the nth element contains the index of the solution
    # where n clusters are found for the first time. If an element is -1, that
    # number of clusters is not found
    cluster_solution_index = rep(-1, nrow(cggm_output$inputs$S))
    for (i in 1:length(cggm_output$cluster_counts)) {
        c = cggm_output$cluster_counts[i]

        if (cluster_solution_index[c] < 0) {
            cluster_solution_index[c] = i
        }
    }
    cggm_output$cluster_solution_index = cluster_solution_index

    # The number of solutions
    cggm_output$n = length(cggm_output$cluster_counts)

    return(cggm_output)
}
