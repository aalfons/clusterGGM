#' Generate Covariance Matrix
#'
#' This function generates a \code{n_vars} by \code{n_vars} covariance matrix in
#' which the rows/columns are genereated from \code{n_clusters} clusters.
#'
#' @param n_vars The number of variables in the covariance matrix.
#' @param n_clusters The number of clusters in the covariance matrix.
#' @param n_draws The number of observations that should be drawn from a
#' multivariate normal distribution with the true covariance matrix.
#' @param shuffle Logical indicating whether to shuffle the covariance matrix.
#' Defaults to \code{FALSE}.
#'
#' @return A list containing the true covariance matrix, the sample covariance
#' matrix, the true cluster labels, and the sample data set generated using the
#' true covariance matrix.
#'
#' @export
generateCovariance <- function(n_vars, n_clusters, n_draws = 100 * n_vars,
                               shuffle = FALSE)
{
    if (n_clusters > n_vars) {
        stop(paste("The number of clusters must be smaller than or equal to",
                   "the number of variables (test failed: c_clusters <=",
                   " n_vars)."))
    }

    if (shuffle) {
        warning("Shuffle is not implemented yet.")
    }

    # Membership vector
    u = c(1:n_clusters)
    u = c(u, sample(u, n_vars - n_clusters, replace = TRUE))
    u = sort(u)

    # Generate clustered data
    Ra = CGGMR:::.generateRA(n_clusters)
    R = Ra$R
    A = Ra$A

    # Compute Theta
    Theta = R[u, u] + diag(A[u])

    # True covariance matrix
    Sigma = solve(Theta)

    # Draw data
    data = mvtnorm::rmvnorm(n_draws, sigma = Sigma)

    # Compute sample covariance matrix
    S = cov(data)

    # Fill result
    result = list()
    result$sample = S
    result$true = Sigma
    result$clusters = u
    result$data = data

    return(result)
}
