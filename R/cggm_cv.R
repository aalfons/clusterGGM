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
#' @param refit Logical, indicating whether the result from \code{\link{cggm}}
#' should be refitted under the constraint of the identified clusters but
#' without additional penalization. See also \code{\link{cggm_refit}}. Defaults
#' to \code{FALSE}.
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
                    scoring_method = "NLL", refit = FALSE, one_se_rule = FALSE,
                    verbose = 0, ...)
{
    # Method for computing the covariance matrix
    cov_method = "pearson"

    # Check whether lambda should be tuned automatically
    auto_lambda = !("lambda" %in% names(tune_grid))

    # Refit can only be done in combination with automatic lambda tuning
    if (refit & !auto_lambda) {
        warning(paste("The option refit = TRUE can only be used with automatic",
                      "lambda tuning. Provided values for lambda are discarded",
                      "and lambda is tuned automatically for each k and phi."))
        tune_grid$lambda = NULL
        auto_lambda = TRUE
    }

    # The one SE rule can only be used in combination with automatic lambda
    # tuning
    if (one_se_rule & !auto_lambda) {
        warning(paste("The option one_se_rule = TRUE can only be used with",
                      "automatic lambda tuning. Provided values for lambda are",
                      "discarded and lambda is tuned automatically for each k",
                      "and phi."))
        tune_grid$lambda = NULL
        auto_lambda = TRUE
    }

    # Create folds for k fold cross validation
    if (is.null(folds)) {
        n = nrow(X)
        folds = cv_folds(n, K = kfold)
    } else {
        kfold = length(folds)
    }

    # Remove duplicate hyperparameter configurations
    tune_grid = unique(tune_grid)

    # Store original tune grid
    tune_grid_og = tune_grid

    # Based on whether lambda is set automatically or user-supplied values are
    # used, do some preparations
    if (auto_lambda) {
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

    # Compute sample covariance matrix based on the complete data set
    S = stats::cov(X, method = cov_method)

    # Perform cross validation
    cv_results = lapply(1:nrow(tune_grid), function(tune_grid_i) {
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
        }

        # Keep track of the scores for for the current combination of k and phi
        scores_mat = matrix(0, nrow = length(lambdas), ncol = kfold)

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
            if (!refit) {
                # Without refitting, there is a one to one match between scores
                # and lambdas for which there is a solution
                for (score_i in 1:nrow(scores_mat)) {
                    scores_mat[score_i, f_i] =
                        CGGMR:::.neg_log_likelihood(S_test,
                                                    get_Theta(res, score_i))
                }
            } else {
                # First, do a refit under the restriction of the obtained
                # clusters without any other penalization
                res_refit = cggm_refit(res, verbose = 0)

                # For each value for lambda for which a cv score has to be
                # obtained, find the largest lambda that is smaller than that
                # for which there is a refit result. This is equivalent to
                # checking the number of clusters in the original result, and
                # finding that number of clusters in the refitted result
                for (score_i in 1:nrow(scores_mat)) {
                    refit_index = res_refit$
                        cluster_solution_index[res$cluster_counts[score_i]]

                    # Compute cv score as before
                    scores_mat[score_i, f_i] =
                        CGGMR:::.neg_log_likelihood(S_test,
                                                    get_Theta(res_refit,
                                                              refit_index))
                }
            }
        }

        # Get the weight of each fold, scores are weighted by this value as a
        # larger test set should carry more weight in the average score
        fold_weights = sapply(1:length(folds), function (i) {
            length(folds[[i]])
        })
        fold_weights = fold_weights / sum(fold_weights)

        # Compute mean scores and standard deviation
        scores = scores_mat %*% fold_weights
        scores_sd = sweep(scores_mat, 1, scores)^2 %*% fold_weights
        scores_sd = sqrt(scores_sd / (length(folds) - 1))

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
            if (!refit) {
                best_index = which.min(scores)

                # Length of the interval for this score for lambda is irrelevant
                lambda_intv_length = 0
            } else {
                ## If there are multiple best values for lambda (multiple values
                ## for lambda yield the same clustering), select the one that is
                ## closest to the midpoint of the longest interval of these
                ## lambdas
                # Get start and stop indices of sequences of lowest scores
                starts = which(diff(c(0L, scores == min(scores))) == 1L)
                stops = which(diff(c(scores == min(scores), 0L)) == -1L)

                # Interval lengths
                interval_lengths = sapply(1:length(starts), function(i) {
                    lambdas[stops[i]] - lambdas[starts[i]]
                })

                # Longest interval
                interval_index = which.max(interval_lengths)

                # Midpoint of longest interval
                mean_best_lambdas =
                    (lambdas[starts[interval_index]] +
                         lambdas[stops[interval_index]]) / 2

                # Index of lambda closest to midpoint of longest interval of
                # lowest cv scores
                best_index = which.min(abs(lambdas - mean_best_lambdas))

                # Length of the interval for this score for lambda is zero
                lambda_intv_length = max(interval_lengths)
            }

            # Select lambda 1se
            max_score = scores[best_index] + scores_sd[best_index] /
                sqrt(length(folds))
            lambda_1se = max(lambdas[scores <= max_score])

            # Print results of the hyperparameter settings
            if (verbose > 0) {
                best_lambda = lambdas[best_index]
                best_score = scores[best_index]

                cat("                  lambda (opt) = ")
                cat(paste(round(best_lambda, digits = 4), "\n", sep = ""))
                cat("                   score (opt) = ")
                cat(paste(round(best_score, digits = 4), "\n", sep = ""))
            }

            return(list(
                res = data.frame(phi = phi, k = k, lambda = lambdas[best_index],
                                 lambda_intv_length = lambda_intv_length,
                                 score = scores[best_index],
                                 lambda_1se = lambda_1se),
                lambdas = lambdas
            ))
        } else {
            # Indices for which current k and phi match the score dataframe
            indices = which(tune_grid_og$k == k & tune_grid_og$phi == phi)

            # Create dataframe with results for these k and phi and requested
            # lambda
            res = tune_grid_og[indices, ]
            res$score = scores[lambdas %in% tune_grid_og$lambda[indices]]

            # Print results of the hyperparameter settings
            if (verbose > 0) {
                best_index = which.min(res$score)
                best_lambda = res$lambda[best_index]
                best_score = res$score[best_index]

                cat("                  lambda (opt) = ")
                cat(paste(round(best_lambda, digits = 4), "\n", sep = ""))
                cat("                   score (opt) = ")
                cat(paste(round(best_score, digits = 4), "\n", sep = ""))
            }

            return(list(res = res, lambdas = lambdas))
        }
    })

    # Gather results
    cv_scores = do.call(rbind, lapply(cv_results, "[[", 1))

    # Select the best hyper parameter settings
    if (!refit) {
        # If it is present, remove the column with the lambda interval lengths
        cv_scores$lambda_intv_length = NULL

        # Select index with lowest score
        best_index = which.min(cv_scores$score)
    } else {
        # When using refit, results that should be equivalent sometimes result
        # in different cv scores due to numerical inaccuracies, this is
        # (partially) mitigated in the next step
        min_score = min(cv_scores$score)

        for (cv_scores_i in 1:nrow(cv_scores)) {
            if (abs(cv_scores[cv_scores_i, "score"] - min_score) < 1e-6) {
                cv_scores[cv_scores_i, "score"] = min_score
            }
        }

        # Sort scores
        cv_scores_sorted =
            dplyr::arrange(cbind(1:nrow(cv_scores), cv_scores),
                           score, dplyr::desc(lambda_intv_length))

        # Select index with lowest score
        best_index = cv_scores_sorted[1, 1]
    }

    # Compute the weight matrix based on the full sample
    W = cggm_weights(S, phi = cv_scores$phi[best_index],
                     k = cv_scores$k[best_index], connected = connected)

    # If lambda is tuned automatically, select the sequence that belongs to the
    # optimal values of k and phi
    if (auto_lambda) {
        lambdas_list = lapply(cv_results, "[[", 2)
        lambdas = lambdas_list[[best_index]]
    }

    # Select index of best performing lambda
    if (!one_se_rule) {
        best_lambda_index = which(cv_scores$lambda[best_index] == lambdas)

        # Remove lambda_1se from the cv scores
        cv_scores$lambda_1se = NULL
    } else {
        best_lambda_index = which(cv_scores$lambda_1se[best_index] == lambdas)
    }

    # Run the algorithm with optimal k and phi for all lambdas
    res = cggm(S = S, W = W, lambda = lambdas, expand = FALSE, ...)

    # Refit if required
    if (refit) {
        res_refit = cggm_refit(res, verbose = 0)

        # Update the best lambda index
        best_lambda_index = res_refit$
            cluster_solution_index[res$cluster_counts[best_lambda_index]]

        # Overwrite the result
        res = res_refit
    }

    # Add cross validation results to the result
    result = list()
    result$final = res
    result$scores = cv_scores
    result$opt_index = best_lambda_index

    # Select optimal parameters and reset index
    if (!one_se_rule) {
        result$opt_tune = cv_scores[
            best_index, which(names(cv_scores) %in% c("k", "phi", "lambda"))
        ]
    } else {
        result$opt_tune = cv_scores[
            best_index, which(names(cv_scores) %in% c("k", "phi", "lambda_1se"))
        ]
        result$opt_tune$lambda = result$opt_tune$lambda_1se
        result$opt_tune$lambda_1se = NULL
    }
    result$opt_tune = data.frame(result$opt_tune, row.names = NULL)
    class(result) = "CGGM_CV"

    return(result)
}
