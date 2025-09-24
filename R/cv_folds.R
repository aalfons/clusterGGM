#' Create Cross-Validation Folds
#'
#' Obtain indices for splitting observations into \eqn{K} blocks to be to be
#' folded into training and test data during \eqn{K}-fold cross-validation.
#'
#' @param n  an integer giving the number of observations to be split.
#' @param K  an integer giving the number of blocks into which the observations
#' should be split (the default is five).
#'
#' @return A list of indices giving the blocks of observations to be folded
#' into training and test data during cross-validation.
#'
#' @author Andreas Alfons
#'
#' @seealso
#' \code{\link{cggm_cv}()}
#'
#' @examples
#' cv_folds(20, K = 5)
#'
#' @export

# Function to set up folds for K-fold cross-validation
cv_folds <- function(n, K = 5L) {
    # permute observations
    indices <- sample.int(n)
    # assign a block to each observation
    blocks <- rep(seq_len(K), length.out = n)
    # split the permuted observations according to the block they belong to
    folds <- split(indices, blocks)
    names(folds) <- NULL
    folds
}
