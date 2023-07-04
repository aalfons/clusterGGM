convertCGGMOutput <- function(cggm_output)
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
    
    for (i in 1:n_results) {
        # Cluster (cumulative) counts for the ith result
        c_i = cggm_output$cluster_counts[i]
        cc_i = cumulative_counts[i]
        
        # Get R and A to construct Theta
        res_R = cggm_output$R[1:c_i, (cc_i + 1):(cc_i + c_i)]
        res_A = cggm_output$A[1:c_i, i]
        res_Theta = computeTheta(as.matrix(res_R), res_A, cggm_output$clusters[, i] - 1)
        
        # Put Theta and the cluster IDs in a list
        res_i = list()
        res_i$A = res_A
        res_i$R = res_R
        res_i$Theta = res_Theta
        res_i$clusters = cggm_output$clusters[, i]
        res_i$lambda = cggm_output$lambdas[i]
        res_i$loss = cggm_output$losses[i]
        
        # Append result
        result[[i]] = res_i
    }
    
    return(result)
}
