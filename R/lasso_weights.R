#' Compute the Weight Matrix for the Lasso Penalty
#'
#' Compute the weight matrix for the lasso penalty in the clusterpath estimator
#' of the Gaussian graphical model (CGGM).
#'
#' @param S The sample covariance matrix of the data.
#' @param unit A logical indicating whether the weights should be all one or
#' based on the inverse of \code{S}.
#'
#' @return A weight matrix for the lasso penalty.
#'
#' @author Daniel J.W. Touw
#'
#' @references
#' D.J.W. Touw, A. Alfons, P.J.F. Groenen and I. Wilms (2025)
#' \emph{Clusterpath Gaussian Graphical Modeling}. arXiv:2407.00644.
#' doi:10.48550/arXiv.2407.00644.
#'
#' @seealso
#' \code{\link{clusterpath_weights}()}, \code{\link{cggm}()},
#' \code{\link{cggm_refit}()}, \code{\link{cggm_cv}()}
#'
#' @example inst/doc/examples/example-lasso_weights.R
#'
#' @export
lasso_weights <- function(S, unit = FALSE)
{
    # Initial estimate for Theta
    Theta = .initial_Theta(S)

    # Compute weights
    if (!unit) {
        weights = 1 / matrix(sapply(abs(Theta), max, 5e-3), nrow = nrow(Theta))
    } else {
        weights = matrix(1, nrow = nrow(Theta), ncol = ncol(Theta))
    }

    # Set diagonal to zero
    weights = weights - diag(diag(weights))

    # Make sure the weights sum to what unit weights would sum up to
    weights = weights / sum(weights) * nrow(weights) * (nrow(weights) - 1)

    return(weights)
}
