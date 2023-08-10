#' Cross Validation for the Clusterpath Gaussian Graphical Model
#'
#' This function performs k-fold cross validation to tune the weight matrix
#' parameters phi and k (for k nearest neighbors, knn) and the regularization
#' parameter lambda.
#'
#' @param X The sample covariance matrix of the data.
#' @param lambdas A numeric vector of tuning parameters for regularization.
#' Make sure the values are monotonically increasing.
#' @param phi A numeric vector of tuning parameters for the weights.
#' @param k A vector of integers for the number of nearest neighbors that
#' should be used to set weights to a nonzero value.
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
cggmCV <- function(X, lambdas, phi, k, kfold = 5, folds = NULL,
                   cov_method = "pearson", gss_tol = 1e-4, conv_tol = 1e-9,
                   fusion_type = "proximity", fusion_threshold = NULL,
                   max_iter = 2000, use_Newton = TRUE)
{
    # Create folds for k fold cross validation
    if (is.null(folds)) {
        n = nrow(X)
        folds = caret::createFolds(1:n, k = kfold)
    } else {
        kfold = length(folds)
    }

    # Create an array to store cross validation scores
    dim1 = paste("lambda=", lambdas, sep = "")
    dim2 = paste("phi=", phi, sep = "")
    dim3 = paste("k=", k, sep = "")
    scores = array(0, dim = c(length(lambdas), length(phi), length(k)),
                   dimnames = list(dim1, dim2, dim3))

    # Do the kfold cross validation
    for (fi in 1:kfold) {
        # Select training and test samples for fold fi
        X.train = X[-folds[[fi]], ]
        X.test = X[folds[[fi]], ]
        S.train = stats::cov(X.train, method = cov_method)
        S.test = stats::cov(X.test, method = cov_method)

        for (phi_i in 1:length(phi)) {
            for (k_i in 1:length(k)) {
                # Compute the weight matrix based on the training sample
                W.train = cggmWeights(S.train, phi = phi[phi_i], k = k[k_i])

                # Run the algorithm
                res = cggm(S.train, W.train, lambdas, gss_tol = gss_tol,
                           conv_tol = conv_tol, fusion_type = fusion_type,
                           fusion_threshold = fusion_threshold,
                           max_iter = max_iter, store_all_res = TRUE,
                           use_Newton = use_Newton, profile = FALSE,
                           verbose = 0)

                # Compute the cross validation scores for this fold
                for (lambda_i in 1:length(lambdas)) {
                    scores[lambda_i, phi_i, k_i] =
                        scores[lambda_i, phi_i, k_i] -
                        log(det(res$Theta[[lambda_i]])) +
                        sum(diag(S.test %*% res$Theta[[lambda_i]]))
                }
            }
        }
    }

    # Average the scores
    scores = scores / kfold

    # Select the best hyper parameter settings
    best = which(scores == min(scores), arr.ind = TRUE)[1, ]

    # Compute the covariance matrix on the full sample
    S = stats::cov(X, method = cov_method)

    # Compute the weight matrix based on the full sample
    W = cggmWeights(S, phi = phi[best[2]], k = k[best[3]])

    # Run the algorithm for all lambdas up to the best one
    res = cggm(S, W, lambdas[1:best[1]], gss_tol = gss_tol,
               conv_tol = conv_tol, fusion_type = fusion_type,
               fusion_threshold = fusion_threshold, max_iter = max_iter,
               store_all_res = TRUE, use_Newton = use_Newton, profile = FALSE,
               verbose = 0)

    # Prepare output
    res$loss = res$losses[best[1]]
    res$losses = NULL
    res$lambda = res$lambdas[best[1]]
    res$lambdas = NULL
    res$phi = phi[best[2]]
    res$k = k[best[3]]
    res$cluster_count = res$cluster_counts[best[1]]
    res$cluster_counts = NULL
    res$Theta = res$Theta[[best[1]]]
    res$R = res$R[[best[1]]]
    res$A = res$A[[best[1]]]
    res$clusters = res$clusters[[best[1]]]
    res$cluster_solution_index = NULL
    res$n = NULL
    res$scores = scores

    return(res)
}
