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
    result_R = list()
    result_A = list()
    result_clusters = list()

    for (i in 1:n_results) {
        # Cluster (cumulative) counts for the ith result
        c_i = cggm_output$cluster_counts[i]
        cc_i = cumulative_counts[i]

        # Append results
        result_A[[i]] = cggm_output$A[1:c_i, i]
        result_R[[i]] = cggm_output$R[1:c_i, (cc_i + 1):(cc_i + c_i)]
    }

    result$losses = cggm_output$losses
    result$lambdas = cggm_output$lambdas
    result$cluster_counts = cggm_output$cluster_counts
    result$R = result_R
    result$A = result_A
    result$clusters = t(cggm_output$clusters)

    return(result)
}
