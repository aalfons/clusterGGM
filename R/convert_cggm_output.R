.convert_cggm_output <- function(cggm_output)
{
    # Number of results to process
    n_results = length(cggm_output$cluster_counts)

    # Get the sum of the counts before the ith count
    cumulative_counts = rep(0, n_results)
    if (n_results > 1) {
        for (i in 2:n_results) {
            cumulative_counts[i] = cumulative_counts[i - 1] +
                cggm_output$cluster_counts[i - 1]
        }
    }

    # List for the processed results
    result = list()

    # Lists of the parts of the solutions
    result_Theta = list()
    result_R = list()
    result_A = list()
    result_clusters = list()

    for (i in 1:n_results) {
        # Cluster (cumulative) counts for the ith result
        c_i = cggm_output$cluster_counts[i]
        cc_i = cumulative_counts[i]

        # Get R and A to construct Theta
        res_R = cggm_output$R[1:c_i, (cc_i + 1):(cc_i + c_i)]
        res_A = cggm_output$A[1:c_i, i]
        res_Theta = .compute_Theta(
            as.matrix(res_R), res_A, cggm_output$clusters[, i] - 1
        )

        # Append results
        result_A[[i]] = res_A
        result_R[[i]] = res_R
        result_Theta[[i]] = res_Theta
    }

    result$losses = cggm_output$losses
    result$lambdas = cggm_output$lambdas
    result$cluster_counts = cggm_output$cluster_counts
    result$Theta = result_Theta
    result$R = result_R
    result$A = result_A
    result$clusters = t(cggm_output$clusters)

    return(result)
}
