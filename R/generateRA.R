.generateRA <- function(K)
{
    # Generate symmetric matrix R
    R = matrix(rnorm(K * K * 2), nrow = K)
    for (i in 1:K) {
        R[i, ] = R[i, ] - mean(R[i, ])
    }
    R = R %*% t(R) / (2 * K)
    
    # Generate vector A with positive elements using a truncated normal 
    # distribution
    A = rnorm(K, mean = 1, sd = 0.5)
    for (i in 1:K) {
        while (A[i] <= 0) {
            A[i] = rnorm(1, mean = 1, sd = 0.5)
        }
    }
    
    return(list("R" = R, "A" = A))
}
