#' @importFrom stats cov
.helper_cv_iteration <- function(
        tune_grid_i, tune_grid, tune_grid_og, auto_lambda, X, S,
        lasso_unit_weights, connected, lambdas, lambdas_init, folds, kfold,
        cov_method, estimate_Sigma, verbose, ...)
{
    ## If necessary, begin with computing a sequence for lambda to be used
    ## for the solution path for the given combination of k and phi.
    # Select k and phi
    k = tune_grid$k[tune_grid_i]
    phi = tune_grid$phi[tune_grid_i]
    lambda_lasso = tune_grid$lambda_lasso[tune_grid_i]

    # Print the current hyperparameter settings
    if (verbose > 0) {
        cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S : "))
        cat(paste("[k, phi, lambda_lasso] = [", k, ", ",
                  round(phi, 4), ", ", round(lambda_lasso, 4), "]\n",
                  sep = ""))
    }

    if (auto_lambda) {
        # Compute weight matrix for the sample covariance matrix based on
        # the complete sample
        W_cpath = clusterpath_weights(
            S = S, phi = phi, k = k, connected = connected
        )

        # Compute weight matrix for the lasso penalty based on the complete
        # sample
        W_lasso = lasso_weights(S, unit = lasso_unit_weights)

        # Compute the solution path, expanding it so that the consecutive
        # solutions for Theta do not differ too much
        res = cggm(
            S = S, W_cpath = W_cpath, W_lasso = W_lasso,
            lambda_cpath = lambdas_init, lambda_lasso = lambda_lasso,
            expand = TRUE, ...
        )

        # Set lambdas
        lambdas = res$lambdas

        # Get the index of lambda for which the minimum number of clusters is
        # first found  and use the sequence up to the corresponding value
        index <- which.min(res$cluster_counts)
        lambdas <- res$lambdas[seq_len(index)]

    }

    # Keep track of the scores for for the current combination of k and phi
    scores_mat_fit = matrix(0, nrow = length(lambdas), ncol = kfold)
    scores_mat_refit = matrix(0, nrow = length(lambdas), ncol = kfold)

    # Do the kfold cross validation
    for (f_i in 1:kfold) {
        # Select training and test samples for fold f_i
        X_train = X[-folds[[f_i]], ]
        X_test = X[folds[[f_i]], ]
        S_train = stats::cov(X_train, method = cov_method)
        S_test = stats::cov(X_test, method = cov_method)

        # If Sigma should be estimated, both covariance matrices should be
        # inverted
        if (estimate_Sigma) {
            S_train = .initial_Theta(S_train)
            S_test = .initial_Theta(S_test)
        }

        # Compute the weight matrix based on the training sample
        W_train = clusterpath_weights(
            S_train, phi = phi, k = k, connected = connected
        )

        # Compute the lasso weight matrix based on the training sample
        W_lasso_train = lasso_weights(S_train, unit = lasso_unit_weights)

        # Run the algorithm
        res_fit = cggm(
            S = S_train, W_cpath = W_train, W_lasso = W_lasso_train,
            lambda_cpath = lambdas, lambda_lasso = lambda_lasso,
            expand = FALSE, ...
        )

        # Without refitting, there is a one to one match between scores and
        # lambdas for which there is a solution
        for (score_i in 1:nrow(scores_mat_fit)) {
            scores_mat_fit[score_i, f_i] = .neg_log_likelihood(
                S_test, get_Theta(res_fit, score_i)
            )
        }

        # Refit the result
        res_refit = cggm_refit(res_fit, verbose = 0)

        # For each value for lambda for which a cv score has to be obtained,
        # find the largest lambda that is smaller than that for which there
        # is a refit result. This is equivalent to checking the number of
        # clusters in the original result, and finding that number of
        # clusters in the refitted result
        for (score_i in 1:nrow(scores_mat_refit)) {
            refit_index = res_refit$
                cluster_solution_index[res_fit$cluster_counts[score_i]]

            # Compute cv score as before
            scores_mat_refit[score_i, f_i] = .neg_log_likelihood(
                S_test, get_Theta(res_refit, refit_index)
            )
        }
    }

    # Get the weight of each fold, scores are weighted by this value as a
    # larger test set should carry more weight in the average score
    fold_weights = sapply(1:length(folds), function (i) {
        length(folds[[i]])
    })
    fold_weights = fold_weights / sum(fold_weights)

    # Compute mean scores and standard deviation
    scores_fit = scores_mat_fit %*% fold_weights
    scores_refit = scores_mat_refit %*% fold_weights

    # If lambda is tuned automatically, select the best performing value to
    # be added to the results. Otherwise, fill in the required values for
    # lambda
    if (auto_lambda) {
        # Best performing value of lambda for the normal fit
        best_index_fit = which.min(scores_fit)

        ## If there are multiple best values for lambda (multiple values for
        ## lambda yield the same clustering), select the one that is closest
        ## to the midpoint of the longest interval of these lambdas
        # Get start and stop indices of sequences of lowest scores
        starts = which(diff(c(0L, scores_refit == min(scores_refit))) == 1L)
        stops = which(diff(c(scores_refit == min(scores_refit), 0L)) == -1L)

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
        best_index_refit = which.min(abs(lambdas - mean_best_lambdas))

        # Length of the interval for this score for lambda is zero
        lambda_intv_length_refit = max(interval_lengths)

        # Save all CV scores for all lambdas
        all_scores_fit = cbind(lambdas, scores_fit[, 1])
        all_scores_refit = cbind(lambdas, scores_refit[, 1])

        return(list(
            res_fit = data.frame(
                phi = phi, k = k, lambda_lasso = lambda_lasso,
                lambda = lambdas[best_index_fit],
                score = scores_fit[best_index_fit]
            ),
            res_refit = data.frame(
                phi = phi, k = k, lambda_lasso = lambda_lasso,
                lambda = lambdas[best_index_refit],
                lambda_intv_length = lambda_intv_length_refit,
                score = scores_refit[best_index_refit]
            ),
            lambdas = lambdas,
            all_scores_fit = all_scores_fit,
            all_scores_refit = all_scores_refit
        ))
    } else {
        # Indices for which current k and phi match the score dataframe
        indices = which(
            tune_grid_og$k == k & tune_grid_og$phi == phi &
                tune_grid_og$lambda_lasso == lambda_lasso
        )

        # Create dataframe with results for these k and phi and requested
        # lambda
        res_fit = tune_grid_og[indices, ]
        res_fit$score = scores_fit[
            lambdas %in% tune_grid_og$lambda[indices]
        ]

        # Do the same for the refitted results
        res_refit = tune_grid_og[indices, ]
        res_refit$score = scores_refit[
            lambdas %in% tune_grid_og$lambda[indices]
        ]

        # Save all CV scores for all lambdas
        all_scores_fit = cbind(lambdas, scores_fit[, 1])
        all_scores_refit = cbind(lambdas, scores_refit[, 1])

        return(list(
            res_fit = res_fit, res_refit = res_refit, lambdas = lambdas,
            all_scores_fit = all_scores_fit, all_scores_refit = all_scores_refit
        ))
    }
}


#' Cross Validation for the Clusterpath Estimator of the Gaussian Graphical Model
#'
#' Perform cross validation to tune the weight matrix parameters \code{phi}
#' and \code{k} (for k-nearest-neighbors) as well as the aggregation parameter
#' \code{lambda_cpath} and the sparsity parameter \code{lambda_lasso} of the
#' clusterpath estimator of the Gaussian Graphical Model (CGGM) in order to
#' obtain a sparse estimate with variable clustering of the precision matrix or
#' the covariance matrix. The scoring metric is the negative log-likelihood
#' (lower is better).
#'
#' @param X The \code{n} times \code{p} matrix holding the data, with \code{n}
#' observations and \code{p} variables.
#' @param tune_grid A data frame with values of the tuning parameters. Each row
#' is a combination of parameters that is evaluated. The columns have the names
#' of the tuning parameters and should include \code{k} and \code{phi}. The
#' sparsity parameter \code{lambda_lasso} and the aggregation parameter
#' \code{lambda} are optional. If there is no column named \code{lambda_lasso},
#' the sparsity parameter is set to 0. If there is no column named
#' \code{lambda}, an appropriate range for the aggregation parameter is
#' selected for each combination of \code{k}, \code{phi}, and
#' \code{lambda_lasso}.
#' @param kfold The number of folds. Defaults to 5.
#' @param folds Optional argument to manually set the folds for the cross
#' validation procedure. If this is not \code{NULL}, it overrides the
#' \code{kfold} argument. Defaults to \code{NULL}.
#' @param connected Logical, indicating whether connectedness of the weight
#' matrix should be ensured. Defaults to \code{TRUE}. See
#' \code{\link{clusterpath_weights}()}.
#' @param fit Logical, indicating whether the cross-validation procedure
#' should consider the result from \code{\link{cggm}()}, before refitting is
#' applied. Defaults to \code{TRUE}. At least one of \code{fit} and \code{refit}
#' should be \code{TRUE}.
#' @param refit Logical, indicating whether the cross-validation procedure
#' should also consider the refitted result from \code{\link{cggm}()}. See also
#' \code{\link{cggm_refit}()}. Defaults to \code{TRUE}. At least one of
#' \code{fit} and \code{refit} should be \code{TRUE}.
#' @param lasso_unit_weights Logical, indicating whether the weights in the
#' sparsity penalty should be all one or decreasing in the magnitude of the
#' corresponding element of the inverse of the sample covariance matrix.
#' Defaults to \code{FALSE}.
#' @param estimate_Sigma Logical, indicating whether CGGM should be used to
#' estimate the covariance matrix based on the sample precision matrix. Defaults
#' to \code{FALSE}.
#' @param verbose Determines the amount of information printed during the
#' cross validation. Defaults to \code{0}.
#' @param n_jobs Number of parallel jobs used for cross validation. If 0 or
#' smaller, uses the maximum available number of physical cores. Defaults to
#' \code{1} (sequential).
#' @param ... Additional arguments to be passed down to \code{\link{cggm}()}
#' and \code{\link{cggm_refit}()}.
#'
#' @return An object of class \code{"CGGM_CV"} with the following components:
#' \item{fit}{A list with cross-validation results for CGGM without the
#' refitting step. It consists of four components:
#' \itemize{
#' \item{\code{final}} (an object of class \code{"CGGM"} corresponding to the
#' final model fit using the optimal values of the tuning parameters; see
#' \code{\link{cggm}()})
#' \item{\code{scores}} (a data frame containing the values of the tuning
#' parameters and the corresponding cross-validation scores)
#' \item{\code{opt_index}} (the index of the optimal aggregation parameter
#' \code{lambda_cpath} in the final model fit)
#' \item{\code{opt_tune}} (a data frame containing the values of the tuning
#' parameters)
#' }}
#' \item{refit}{A list with cross-validation results for CGGM including the
#' refitting step. It contains the same four components as above, except that
#' \code{final} is an object of class \code{"CGGM_refit"} (see
#' \code{\link{cggm_refit}()}).}
#' \item{raw_cv_results}{A list of raw cross-validation results before
#' restructuring.}
#' \item{best}{A character string indicating whether the optimal model fit
#' without the refitting step (\code{"fit"}) or including the refitting step
#' ("refit") has a better cross-validation score.}
#'
#' @note The function interface and output structure are still experimental and
#' may change in the next version.
#'
#' @author Daniel J.W. Touw, modifications by Andreas Alfons
#'
#' @references
#' D.J.W. Touw, A. Alfons, P.J.F. Groenen and I. Wilms (2025)
#' \emph{Clusterpath Gaussian Graphical Modeling}. arXiv:2407.00644.
#' doi:10.48550/arXiv.2407.00644.
#'
#' @seealso
#' \code{\link{clusterpath_weights}()}, \code{\link{lasso_weights}()},
#' \code{\link{cggm}()}, \code{\link{cggm_refit}()}
#'
#' @examples
#' \donttest{
#' # Generate data
#' set.seed(3)
#' Theta <- matrix(
#'   c(2, 1, 0, 0,
#'     1, 2, 0, 0,
#'     0, 0, 4, 1,
#'     0, 0, 1, 4),
#'   nrow = 4
#' )
#' X <- mvtnorm::rmvnorm(n = 100, sigma = solve(Theta))
#'
#' # Use cross-validation to select the tuning parameters
#' fit_cv <- cggm_cv(
#'   X = X,
#'   tune_grid = expand.grid(
#'     phi = 1,
#'     k = 2,
#'     lambda_lasso = c(0, 0.02),
#'     lambda = seq(0, 0.2, by = 0.01)
#'   ),
#'   folds = cv_folds(nrow(X), 5)
#' )
#'
#' # The best solution has 2 clusters
#' get_Theta(fit_cv)
#' get_clusters(fit_cv)
#' }
#'
#' @importFrom dplyr arrange desc
#' @importFrom parallel clusterExport detectCores makePSOCKcluster parLapply stopCluster
#' @importFrom rlang .data
#' @importFrom stats cov
#' @export
cggm_cv <- function(X, tune_grid, kfold = 5, folds = NULL, connected = TRUE,
                    fit = TRUE, refit = TRUE, lasso_unit_weights = FALSE,
                    estimate_Sigma = FALSE, verbose = 0, n_jobs = 1, ...)
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

    # Check if performing cross validation to estimate Sigma is possible
    if (estimate_Sigma) {
        if (min(sapply(folds, length)) <= ncol(X)) {
            stop(paste("The smallest fold does not contain enough data to",
                       "compute the sample precision matrix."))
        }
    }

    # Check if the lasso penalty needs to be tuned
    if (is.null(tune_grid$lambda_lasso)) {
        tune_grid$lambda_lasso = 0
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

        # Initialize the lambdas vector to be empty
        lambdas = c()
    } else {
        # Lambdas is set as all unique values supplied by the user.
        lambdas = unique(c(0, tune_grid$lambda))
        lambdas = sort(lambdas)

        # Make sure the jumps in lambdas are not too large, so expand the vector
        lambdas = .expand_vector(lambdas, 0.01)

        # Remove the colum lambda from tune_grid
        tune_grid$lambda = NULL

        # Select unique rows from tune_grid
        tune_grid = unique(tune_grid)

        # Initialize the lambdas_init vector to be empty
        lambdas_init = c()
    }

    # Compute sample covariance matrix based on the complete data set
    S = stats::cov(X, method = cov_method)

    # If Sigma should be estimated, S takes on the role of sample precision
    # matrix
    if (estimate_Sigma) {
        S = .initial_Theta(S)
    }

    if (n_jobs == 1) {
        # Perform cross validation
        cv_results = lapply(1:nrow(tune_grid), function(tune_grid_i) {
            .helper_cv_iteration(
                tune_grid_i, tune_grid, tune_grid_og, auto_lambda, X, S,
                lasso_unit_weights, connected, lambdas, lambdas_init, folds,
                kfold, cov_method, estimate_Sigma, verbose, ...
            )
        })
    } else {
        ## Parallel cross validation
        # Number of cores to use
        if (n_jobs <= 0) {
            n_cores = parallel::detectCores(logical = FALSE)
        } else {
            n_cores = min(n_jobs, parallel::detectCores(logical = FALSE))
        }

        # Make cluster
        cl = parallel::makePSOCKcluster(
            min(n_cores, parallel::detectCores(logical = FALSE))
        )

        # Export variables and functions to the workers
        parallel::clusterExport(
            cl, ls(envir = environment()), envir = environment()
        )

        cv_results = parallel::parLapply(cl, 1:nrow(tune_grid), function(
            tune_grid_i
        ) {
            .helper_cv_iteration(
                 tune_grid_i, tune_grid, tune_grid_og, auto_lambda, X, S,
                 lasso_unit_weights, connected, lambdas, lambdas_init, folds,
                 kfold, cov_method, estimate_Sigma, 0, ...
            )
        })

        # Stop cluster
        parallel::stopCluster(cl)
    }

    # Create a list with the raw results from the cross validation
    raw_results = list()
    for (i in 1:length(cv_results)) {
        raw_results[[i]] = list()
        raw_results[[i]]$phi = cv_results[[i]]$res_fit$phi[1]
        raw_results[[i]]$k = cv_results[[i]]$res_fit$k[1]
        raw_results[[i]]$lambda_lasso = cv_results[[i]]$res_fit$lambda_lasso[1]
        raw_results[[i]]$all_scores_fit = cv_results[[i]]$all_scores_fit
        raw_results[[i]]$all_scores_refit = cv_results[[i]]$all_scores_refit

        colnames(raw_results[[i]]$all_scores_fit) = c("lambda", "score")
        colnames(raw_results[[i]]$all_scores_refit) = c("lambda", "score")
    }

    # Gather results
    cv_scores_fit = do.call(rbind, lapply(cv_results, "[[", 1))
    cv_scores_refit = do.call(rbind, lapply(cv_results, "[[", 2))

    # Best setting for cggm without refit
    best_index_fit = best_index = which.min(cv_scores_fit$score)

    # When using refit, results that should be equivalent sometimes result in
    # different cv scores due to numerical inaccuracies, this is (partially)
    # mitigated in the next step
    min_score = min(cv_scores_refit$score)

    # TODO: test with relative difference instead
    for (cv_scores_i in 1:nrow(cv_scores_refit)) {
        if (abs(cv_scores_refit[cv_scores_i, "score"] - min_score) < 1e-6) {
            cv_scores_refit[cv_scores_i, "score"] = min_score
        }
    }

    # Sort scores
    if (!is.null(cv_scores_refit$lambda_intv_length)) {
        cv_scores_sorted =
            dplyr::arrange(
                cbind(1:nrow(cv_scores_refit), cv_scores_refit), .data$score,
                dplyr::desc(.data$lambda_intv_length)
            )

        # Select index with lowest score
        best_index_refit = cv_scores_sorted[1, 1]
    } else {
        # Sort scores
        cv_scores_sorted =
            dplyr::arrange(
                cbind(1:nrow(cv_scores_refit), cv_scores_refit), .data$score
            )

        # For multiple scores that are the same for different values for lambda,
        # combine these into an "optimal" value for lambda that is the midpoint
        # of those that all attained the same score
        indices = c()
        for (i in 1:nrow(cv_scores_sorted)) {
            if (all(cv_scores_sorted[i, c("phi", "k", "lambda_lasso", "score")] ==
                    cv_scores_sorted[1, c("phi", "k", "lambda_lasso", "score")])) {
                indices = c(indices, i)
            }
        }

        # Get the midpoint
        opt_lambda = mean(range(cv_scores_sorted[indices, "lambda"]))

        # Finally, select the index with the lambda closest to the midpoint
        best_index_refit = which.min(
            abs(cv_scores_sorted[indices, "lambda"] - opt_lambda)
        )
        best_index_refit = cv_scores_sorted[best_index_refit, 1]
    }

    ## Train the best model without refitting step
    # Compute the weight matrices based on the full sample
    W_cpath = clusterpath_weights(
        S, phi = cv_scores_fit$phi[best_index_fit],
        k = cv_scores_fit$k[best_index_fit], connected = connected
    )
    W_lasso = lasso_weights(S, unit = lasso_unit_weights)

    # If lambda is tuned automatically, select the sequence that belongs to the
    # optimal values of k and phi
    if (auto_lambda) {
        lambdas_list = lapply(cv_results, "[[", 3)
        lambdas = lambdas_list[[best_index_fit]]
    }

    # Run the algorithm with optimal k and phi for all lambdas
    res_fit = cggm(
        S = S, W_cpath = W_cpath, W_lasso = W_lasso, lambda_cpath = lambdas,
        lambda_lasso = cv_scores_fit$lambda_lasso[best_index_fit],
        expand = FALSE, ...
    )

    ## Train the best model with refitting step
    # Compute the weight matrix based on the full sample
    W_cpath = clusterpath_weights(
        S, phi = cv_scores_refit$phi[best_index_refit],
        k = cv_scores_refit$k[best_index_refit], connected = connected
    )

    # If lambda is tuned automatically, select the sequence that belongs to the
    # optimal values of k and phi
    if (auto_lambda) {
        lambdas = lambdas_list[[best_index_refit]]
    }

    # Run the algorithm with optimal k and phi for all lambdas
    res_refit = cggm(
        S = S, W_cpath = W_cpath, W_lasso = W_lasso, lambda_cpath = lambdas,
        lambda_lasso = cv_scores_refit$lambda_lasso[best_index_refit],
        expand = FALSE, ...
    )
    res_refit = cggm_refit(res_refit, verbose = 0)

    # Get indices of the best performing lambdas
    best_lambda_index_fit = which(
        res_fit$lambdas == cv_scores_fit$lambda[best_index_fit]
    )
    best_lambda_index_refit = which.max(res_refit$lambdas[
        res_refit$lambdas <= cv_scores_refit$lambda[best_index_refit]
    ])

    # Prepare results without refitting step
    result_fit = list()
    result_fit$final = res_fit
    result_fit$scores = cv_scores_fit
    result_fit$opt_index = best_lambda_index_fit
    result_fit$opt_tune = cv_scores_fit[
        best_index_fit, c("k", "phi", "lambda_lasso", "lambda")
    ]
    result_fit$opt_tune = data.frame(result_fit$opt_tune, row.names = NULL)

    # Prepare results with refitting step
    result_refit = list()
    result_refit$final = res_refit
    result_refit$scores = cv_scores_refit
    result_refit$scores$lambda_intv_length = NULL
    result_refit$opt_index = best_lambda_index_refit
    result_refit$opt_tune = cv_scores_refit[
        best_index_refit, c("k", "phi", "lambda_lasso", "lambda")
    ]
    result_refit$opt_tune = data.frame(result_refit$opt_tune, row.names = NULL)

    # Combine into one result
    result = list()
    result$fit = result_fit
    result$refit = result_refit
    result$raw_cv_results = raw_results

    # Select which results should be accessed by the accessor functions
    if (fit & !refit) {
            result$best = "fit"
    } else if (!fit & refit) {
            result$best = "refit"
    } else if (fit & refit) {
        if (min(cv_scores_fit$score) < min(cv_scores_refit$score)) {
            result$best = "fit"
        } else {
            result$best = "refit"
        }
    }

    # Set class
    class(result) = "CGGM_CV"

    return(result)
}
