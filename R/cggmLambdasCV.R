#' Cross Validation for the Clusterpath Gaussian Graphical Model
#'
#' This function performs k-fold cross validation to select the optimal value of
#' lambda.
#'
#' @param X The sample covariance matrix of the data.
#' @param lambdas A numeric vector of tuning parameters for regularization.
#' Make sure the values are monotonically increasing.
#' @param phi Tuning parameter of the weights.
#' @param k The number of nearest neighbors that should be used to set weights
#' to a nonzero value. If \code{0 < k < ncol(S)}, the dense weight matrix will
#' be made sparse, otherwise the dense matrix is computed.
#' @param kfold The number of folds. Defaults to 5.
#' @param folds Optional argument to manually set the folds for the cross
#' validation procedure. If this is not \code{NULL}, it overrides the
#' \code{kfold} argument. Defaults to \code{NULL}.
#' @param cov_method Character string indicating which correlation coefficient
#' (or covariance) is to be computed. One of \code{"pearson"}, \code{"kendall"},
#' or \code{"spearman"}: can be abbreviated. Defaults to \code{"pearson"}.
#' @param gss_tol The tolerance value used in the Golden Section Search (GSS)
#' algorithm. Defaults to \code{1e-4}.
#' @param conv_tol The tolerance used to determine convergence. Defaults to
#' \code{1e-9}.
#' @param fusion_type The type of fusion to be used for determining fusion
#' between two objects. Possible values: "proximity" (based on closeness) or
#' one of {"a0", "a1", "a2"} (based on analytical evaluation). Defaults to
#' "proximity".
#' @param fusion_threshold If \code{fusion_type = "proximity"}, it is the
#' threshold for fusing two clusters. For the analytical fusions, it is the
#' threshold below which the analytical check is executed. If NULL, defaults
#' to \code{sqrt(nrow(S)) * 1e-4} times the median distance between the rows of
#' \code{solve(S)} for proximity based fusions and to \code{sqrt(nrow(S))} times
#' the median distance for analytical fusions.
#' @param max_iter The maximum number of iterations allowed for the optimization
#' algorithm. Defaults to \code{5000}.
#' @param use_Newton Logical, indicating whether to use Newton's method in the
#' optimization algorithm. Defaults to \code{TRUE}.
#'
#' @return A list containing the estimated parameters of the CGGM model.
#'
#' @examples
#' # Example usage:
#'
#' @export
cggmLambdasCV <- function(X, lambdas, phi, k, kfold = 5, folds = NULL,
                          cov_method = "pearson", gss_tol = 1e-4,
                          conv_tol = 1e-9, fusion_type = "proximity",
                          fusion_threshold = NULL, max_iter = 2000,
                          use_Newton = TRUE)
{
    warning("This function is obsolete and may be removed in the future.")

    # Create folds for k fold cross validation
    if (is.null(folds)) {
        n = nrow(X)
        folds = cv_folds(n, K = kfold)
    } else {
        kfold = length(folds)
    }

    # Create a matrix to store cross validation scores
    scores = matrix(0, nrow = length(lambdas), ncol = 2)
    scores[, 1] = lambdas
    colnames(scores) = c("lambda", "score")

    for (fi in 1:kfold) {
        # Select training and test samples for fold fi
        X.train = X[-folds[[fi]], ]
        X.test = X[folds[[fi]], ]
        S.train = stats::cov(X.train, method = cov_method)
        S.test = stats::cov(X.test, method = cov_method)

        # Compute the weight matrix based on the training sample
        W.train = cggmWeights(S.train, phi = phi, k = k)

        # Run the algorithm
        res = cggm(S.train, W.train, lambdas, gss_tol = gss_tol,
                   conv_tol = conv_tol, fusion_type = fusion_type,
                   fusion_threshold = fusion_threshold, max_iter = max_iter,
                   store_all_res = TRUE, use_Newton = use_Newton,
                   profile = FALSE, verbose = 0)

        # Compute the cross validation scores for this fold
        for (si in 1:length(lambdas)) {
            scores[si, 2] = scores[si, 2] -
                log(det(res$Theta[[si]])) +
                sum(diag(S.test %*% res$Theta[[si]]))
        }
    }

    # Average the scores
    scores[, 2] = scores[, 2] / kfold

    # Select the best value for lambda
    best_index = which.min(scores[, 2])

    # Compute the covariance matrix on the full sample
    S = stats::cov(X, method = cov_method)

    # Compute the weight matrix based on the full sample
    W = cggmWeights(S, phi = phi, k = k)

    # Run the algorithm for all lambdas up to the best one
    res = cggm(S, W, lambdas[1:best_index], gss_tol = gss_tol,
               conv_tol = conv_tol, fusion_type = fusion_type,
               fusion_threshold = fusion_threshold, max_iter = max_iter,
               store_all_res = TRUE, use_Newton = use_Newton, profile = FALSE,
               verbose = 0)

    # Prepare output
    res$loss = res$losses[best_index]
    res$losses = NULL
    res$lambda = res$lambdas[best_index]
    res$lambdas = NULL
    res$cluster_count = res$cluster_counts[best_index]
    res$cluster_counts = NULL
    res$Theta = res$Theta[[best_index]]
    res$R = res$R[[best_index]]
    res$A = res$A[[best_index]]
    res$clusters = res$clusters[[best_index]]
    res$cluster_solution_index = NULL
    res$n = NULL
    res$scores = scores

    return(res)
}
