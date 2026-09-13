# Relative difference between the solutions of consecutive values of the
# aggregation parameter, used to decide where the sequence of values for the
# aggregation parameter needs to be refined. The differences are measured on
# the solutions of the minimization, so no sparsity threshold is applied. Only
# two solutions are reconstructed at a time
.consecutive_differences <- function(cggm_output)
{
    diff_norms = rep(0, cggm_output$n - 1)
    Theta_previous = .get_estimate(cggm_output, 1, threshold = FALSE)

    for (i in 2:cggm_output$n) {
        Theta_current = .get_estimate(cggm_output, i, threshold = FALSE)

        diff_norms[i - 1] = norm(Theta_previous - Theta_current, "F") /
            norm(Theta_previous, "F")

        Theta_previous = Theta_current
    }

    return(diff_norms)
}
