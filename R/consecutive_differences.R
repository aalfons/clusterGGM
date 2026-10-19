# Relative difference between the solutions of consecutive values of the
# aggregation parameter, used to decide where the sequence of values for the
# aggregation parameter needs to be refined. The differences are measured on
# the solutions of the minimization, so no sparsity threshold is applied. Only
# two solutions are reconstructed at a time
.consecutive_differences <- function(cggm_output)
{
    diff_norms = rep(0, cggm_output$n - 1)
    Theta_previous = .get_Theta(cggm_output, 1)

    for (i in 2:cggm_output$n) {
        Theta_current = .get_Theta(cggm_output, i)

        diff_norms[i - 1] = norm(Theta_previous - Theta_current, "F") /
            norm(Theta_previous, "F")

        Theta_previous = Theta_current
    }

    return(diff_norms)
}

# Barebones function to extract solution as a matrix. This is called before
# the object is restructured in function cggm(), where the object still stores
# the matrices directly in components 'R' and 'A', as returned by C++.
.get_Theta <- function(object, index) {
  .compute_Theta(as.matrix(object$R[[index]]), object$A[[index]],
                 object$clusters[index, ] - 1)
}
