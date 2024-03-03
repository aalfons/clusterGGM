#' Cross Validation for the Clusterpath Gaussian Graphical Model
#'
#' This function performs k-fold cross validation to tune the weight matrix
#' parameters phi and k (for k nearest neighbors, knn) and the regularization
#' parameter lambda.
#'
#' @param X The sample covariance matrix of the data.
#' @param tune_grid A data frame with values of the tuning parameters. Each row
#' is a combination of parameters that is evaluated. The columns have the names
#' of the tuning parameters and should include \code{k} and \code{phi}. The
#' regularization parameter \code{lambda} is optional. If there is no column
#' named \code{lambda}, an appropriate range is selected for each combination of
#' \code{k} and \code{phi}.
#' @param kfold The number of folds. Defaults to 5.
#' @param folds Optional argument to manually set the folds for the cross
#' validation procedure. If this is not \code{NULL}, it overrides the
#' \code{kfold} argument. Defaults to \code{NULL}.
#' @param connected Logical, indicating whether connectedness of the weight
#' matrix should be ensured. Defaults to \code{TRUE}. See
#' \code{\link{cggm_weights}}.
#' @param scoring_method Method to use for the cross validation scores.
#' Currently, the only choice is \code{NLL} (negative log-likelihood).
#' @param verbose Determines the amount of information printed during the
#' cross validation. Defaults to \code{0}.
#' @param ... Additional arguments meant for \code{\link{cggm}}.
#'
#' @return A list containing the estimated parameters of the CGGM model.
#'
#' @examples
#' # Example usage:
#'
#' @seealso \code{\link{cggm_weights}}, \code{\link{cggm}}
#'
#' @export
cggm_cv <- function(X, tune_grid, kfold = 5, folds = NULL, connected = TRUE,
                    scoring_method = "NLL", verbose = 0, ...)
{
    # Method for computing the covariance matrix
    cov_method = "pearson"

    # Check whether lambda should be tuned automatically
    auto_lambda = !("lambda" %in% names(tune_grid))

    # Create folds for k fold cross validation
    if (is.null(folds)) {
        n = nrow(X)
        folds = cv_folds(n, K = kfold)
    } else {
        kfold = length(folds)
    }

    # Remove duplicate hyperparameter configurations
    tune_grid = unique(tune_grid)

    # Data frame to store results in
    cv_scores = tune_grid

    # Based on whether lambda is set automatically or user-supplied values are
    # used, do some preparations
    if (auto_lambda) {
        # If the user did not supply a tune_grid for lambda, add a column for
        # lambda
        cv_scores$lambda = 0

        # Necessary list when lambda is tuned automatically. Keeps track of the
        # selected lambdas for each combination of k and phi
        lambdas_list = list()

        # Initial lambdas. This sequence will be expanded to appropriate values
        # during the cross validation process
        lambdas_init = c(seq(0, 0.1, 0.01),
                         seq(0.125, 0.25, 0.025),
                         seq(0.3, 0.5, 0.05))
    } else {
        # Lambdas is set as all unique values supplied by the user.
        lambdas = unique(c(0, tune_grid$lambda))
        lambdas = sort(lambdas)

        # Make sure the jumps in lambdas are not too large, so expand the vector
        lambdas = CGGMR:::.expand_vector(lambdas, 0.01)

        # Remove the colum lambda from tune_grid
        tune_grid$lambda = NULL

        # Select unique rows from tune_grid
        tune_grid = unique(tune_grid)
    }

    # Add a column for the scores
    cv_scores$score = 0

    # Compute sample covariance matrix based on the complete data set
    S = stats::cov(X, method = cov_method)

    for (tune_grid_i in 1:nrow(tune_grid)) {
        ## If necessary, begin with computing a sequence for lambda to be used
        ## for the solution path for the given combination of k and phi.
        # Select k and phi
        k = tune_grid$k[tune_grid_i]
        phi = tune_grid$phi[tune_grid_i]

        if (auto_lambda) {
            # Compute weight matrix for the sample covariance matrix based on
            # the complete sample
            W = cggm_weights(S = S, phi = phi, k = k, connected = connected)

            # Compute the solution path, expanding it so that the consecutive
            # solutions for Theta do not differ too much
            res = cggm(S = S, W = W, lambda = lambdas_init, expand = TRUE, ...)

            # Set lambdas
            lambdas = res$lambdas

            # Store lambdas for later, required when training the final model on
            # the tuned hyperparameters
            lambdas_list[[tune_grid_i]] = lambdas
        }

        # Keep track of the scores for for the current combination of k and phi
        scores = rep(0, length(lambdas))

        # Do the kfold cross validation
        for (f_i in 1:kfold) {
            # Select training and test samples for fold f_i
            X_train = X[-folds[[f_i]], ]
            X_test = X[folds[[f_i]], ]
            S_train = stats::cov(X_train, method = cov_method)
            S_test = stats::cov(X_test, method = cov_method)

            # Compute the weight matrix based on the training sample
            W_train = cggm_weights(S_train, phi = phi, k = k,
                                   connected = connected)

            # Run the algorithm
            res = cggm(S = S_train, W = W_train, lambda = lambdas,
                       expand = FALSE, ...)

            # Compute the cross validation score for each lambda
            for (score_i in 1:length(scores)) {
                scores[score_i] = scores[score_i] + nrow(X_test) / nrow(X) *
                    CGGMR:::.neg_log_likelihood(S_test, get_Theta(res, score_i))
            }
        }

        # Print results of the hyperparameter settings
        if (verbose > 0) {
            cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S : "))
            cat(paste("[k, phi] = [", k, ", ", round(phi, digits = 4), "]\n",
                      sep = ""))
        }

        # If lambda is tuned automatically, select the best performing value to
        # be added to the results. Otherwise, fill in the required values for
        # lambda
        if (auto_lambda) {
            # Best performing value of lambda
            best_index = which.min(scores)

            # Fill in lambda and corresponding score
            cv_scores$lambda[tune_grid_i] = lambdas[best_index]
            cv_scores$score[tune_grid_i] = scores[best_index]

            # Print results of the hyperparameter settings
            if (verbose > 0) {
                best_lambda = lambdas[best_index]
                best_score = scores[best_index]

                cat("                  lambda (opt) = ")
                cat(paste(round(best_lambda, digits = 4), "\n", sep = ""))
                cat("                   score (opt) = ")
                cat(paste(round(best_score, digits = 4), "\n", sep = ""))
            }
        } else {
            # Indices for which current k and phi match the score dataframe
            indices = which(cv_scores$k == k & cv_scores$phi == phi)

            # Select the scores for the lambdas present in the grid
            cv_scores[indices, ]$score =
                scores[lambdas %in% cv_scores$lambda[indices]]

            # Print results of the hyperparameter settings
            if (verbose > 0) {
                scores_subset = cv_scores[indices, ]
                best_index = which.min(scores_subset$score)
                best_lambda = scores_subset$lambda[best_index]
                best_score = scores_subset$score[best_index]

                cat("                  lambda (opt) = ")
                cat(paste(round(best_lambda, digits = 4), "\n", sep = ""))
                cat("                   score (opt) = ")
                cat(paste(round(best_score, digits = 4), "\n", sep = ""))
            }
        }
    }

    # Select the best hyper parameter settings
    best_index = which.min(cv_scores$score)

    # Compute the weight matrix based on the full sample
    W = cggm_weights(S, phi = cv_scores$phi[best_index],
                     k = cv_scores$k[best_index], connected = connected)

    # If lambda is tuned automatically, select the sequence that belongs to the
    # optimal values of k and phi
    if (auto_lambda) {
        lambdas = lambdas_list[[best_index]]
    }

    # Select index of best performing lambda
    best_lambda_index = which(cv_scores$lambda[best_index] == lambdas)

    # Run the algorithm with optimal k and phi for all lambdas
    res = cggm(S = S, W = W, lambda = lambdas, expand = FALSE, ...)

    # Add cross validation results to the result
    result = list()
    result$final = res
    result$scores = cv_scores
    result$opt_index = best_lambda_index

    # Select optimal parameters and reset index
    result$opt_tune = cv_scores[best_index, -which(names(cv_scores) == "score")]
    result$opt_tune = data.frame(result$opt_tune, row.names = NULL)
    class(result) = "CGGM_CV"

    return(result)
}
