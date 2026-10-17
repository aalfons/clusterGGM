#' @title Construct a grid of values for a regularization parameter
#'
#' @description Construct a grid of values for a regularization parameter in
#' the sparse clusterpath estimator of the Gaussian graphical model (CGGM) or
#' the tree-aggregated graphical lasso (tag-lasso) estimator, in which each
#' step is increasing or decreasing according to a given multiplication factor
#' compared to the previous step.
#'
#' @param bounds  a numeric vector of length two giving the smallest and
#' largest value of the regularization parameter.
#' @param n_values  a positive integer giving the number of values for the
#' regularization parameter to be generated.
#' @param factor  numeric; multiplication factor larger than 0 to be used to
#' construct the values of the regularization parameter. That is, each step is
#' increasing or decreasing according to this multiplication factor compared to
#' the previous step
#'
#' @return A numeric vector of values for a regularization parameter.
#'
#' @author Andreas Alfons
#'
#' @seealso \code{\link{cggm}()}, \code{\link{cggm_refit}()},
#' \code{\link{cggm_cv}()},
#'
#' \code{\link{taglasso}()}, \code{\link{cv_taglasso}()}
#'
#' @examples
#' # increasing values and step size
#' lambda_grid(c(0, 1), factor = 2)
#'
#' # decreasing values and step size
#' lambda_grid(c(1, 0), factor = 0.5)
#'
#' @export
lambda_grid <- function(bounds, n_values = 10, factor = 1) {
  if (factor == 1) grid <- seq(bounds[1], bounds[2], length.out = n_values)
  else {
    # obtain denominator for base length
    if (factor == 2) denominator <- 2^(n_values-1) - 1
    else denominator <- sum(factor^seq(0, n_values-2))
    # compute base length
    l <- diff(bounds) / denominator
    # initialize grid
    grid <- c(bounds[1], rep.int(NA_real_, n_values-2), bounds[2])
    for (i in seq(2, n_values-1)) {
      grid[i] <- grid[i-1] + factor^(i-2)*l
    }
  }
  # return grid of values
  grid
}
