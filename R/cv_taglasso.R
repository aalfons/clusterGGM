
# Function to evaluate tag-lasso on cross-validation folds
evaluate_taglasso <- function(X, A, lambda1, lambda2, folds, ...) {

  # initializations
  K <- length(folds)
  score <- 0

  # compute score in hold-out samples
  for (k in seq_len(K)) {
    # indices of hold-out sample
    i <- folds[[k]]

    # split data into estimation and hold-out sample
    X_estimation <- X[-i, , drop = FALSE]
    X_holdout <- X[i, , drop = FALSE]

    # compute covariance matrices on estimation and hold-out sample
    S_estimation <- cov(X_estimation)
    S_holdout <- cov(X_holdout)

    # fit method on the estimation sample
    fit <- taglasso(X_estimation, A, lambda1 = lambda1, lambda2 = lambda2, ...)

    # compute score on hold-out sample
    score <- score + .neg_log_likelihood(S_holdout, fit$omega_full)

  }

  # return average score over the hold-out samples
  score / K

}


## Function to select the optimal tuning parameters for taglasso
#' Cross-Validation for the Tree-Aggregated Graphical Lasso
#'
#' Perform cross-validation to tune the aggregation parameter \code{lambda1}
#' and the sparsity parameter \code{lambda2} of the tree-aggregated graphical
#' lasso (tag-lasso) estimator of the precision matrix. The scoring metric is
#' the negative log-likelihood (lower is better).
#'
#' @inheritParams taglasso
#' @param lambda1  a numeric vector of candidate values for the aggregation
#' parameter.
#' @param lambda2 a numeric vector of candidate values for the sparsity
#' parameter.
#' @param folds  a list of indices giving the blocks of observations to be
#' folded into training and test data during cross-validation, as returned by
#' \code{\link{cv_folds}()}.
#' @param \dots additional arguments to be passed down to
#' \code{\link{taglasso}()}.
#'
#' @return A list with the following components:
#' \item{scores}{A data frame containing the values of the tuning parameters
#' and the corresponding cross-validation scores.}
#' \item{lambda1_opt}{The optimal value of the aggregation parameter.}
#' \item{lambda2_opt}{The optimal value of the sparsity parameter.}
#' \item{score_opt}{The cross-validation score of the optimal combination of
#' tuning parameters.}
#'
#' @note The function interface and output structure are still experimental and
#' may change in the next version.
#'
#' @author Andreas Alfons
#'
#' @references
#' I. Wilms and J. Bien (2022) Tree-based Node Aggregation in Sparse Graphical
#' Models. \emph{Journal of Machine Learning Research}, \bold{23}(243), 1--36.
#' https://jmlr.org/papers/v23/21-0105.html
#'
#' @seealso \code{\link{taglasso}()}
#'
#' @examples
#' # Generate data
#' set.seed(3)
#' Theta <- matrix(
#'   c( 1.5, -0.5, -0.5,    0,    0,
#'     -0.5,  1.5, -0.5,    0,    0,
#'     -0.5, -0.5,  1.5,    0,    0,
#'      0  ,  0  ,  0  ,  4/3, -2/3,
#'      0,    0,    0,   -2/3,  4/3),
#'   nrow = 5, ncol = 5
#' )
#' X <- mvtnorm::rmvnorm(n = 100, sigma = solve(Theta))
#'
#' # Realistic tree structure that contains the true solution
#' A <- cbind(
#'   diag(5), c(0, 1, 1, 0, 0), c(0, 0, 0, 1, 1),
#'   c(1, 1, 1, 0, 0), rep(1, 5)
#' )
#'
#' # Perform cross-validation
#' fit_cv <- cv_taglasso(X, A, lambda1 = seq(0, 0.3, by = 0.1),
#'                       lambda2 = seq(0, 0.3, by = 0.1),
#'                       folds = cv_folds(nrow(X), 5))
#'
#' # Estimate the precision matrix for fixed values of the
#' # aggregation and sparsity parameters
#' fit <- taglasso(X, A, lambda1 = fit_cv$lambda1_opt,
#'                 lambda2 = fit_cv$lambda2_opt)
#'
#' # Extract precision matrix and cluster membership
#' round(fit$omega_full, digits = 5)
#' fit$cluster
#'
#' @export
cv_taglasso <- function(X, A, lambda1, lambda2, folds, ...) {

  # define grid of hyperparameters
  grid <- expand.grid(lambda1 = lambda1, lambda2 = lambda2)

  # compute scores
  scores <- mapply(function(lambda1, lambda2, ...) {
    evaluate_taglasso(X, A, lambda1 = lambda1, lambda2 = lambda2,
                      folds = folds, ...)
  }, lambda1 = grid$lambda1, lambda2 = grid$lambda2, ...)

  # Select best performing hyperparameters
  select <- which.min(scores)
  lambda1_opt <- grid[select, 1]
  lambda2_opt <- grid[select, 2]

  # Return list of best performing hyperparameters
  list(scores = cbind(grid, score = scores), lambda1_opt = lambda1_opt,
       lambda2_opt = lambda2_opt, score_opt = scores[select])

}
