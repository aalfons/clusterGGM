cggm_cv2 <- function(X, grid, kfold = 5, folds = NULL, connected = FALSE,
                     scoring_method = "NLL", ...)
{
    # Method for computing the covariance matrix
    cov_method = "pearson"

    # Check whether lambda should be tuned automatically
    auto_lambda = !("lambda" %in% names(grid))

    # Create folds for k fold cross validation
    if (is.null(folds)) {
        n = nrow(X)
        folds = cv_folds(n, K = kfold)
    } else {
        kfold = length(folds)
    }

    # Data frame to store results in
    cv_scores = grid

    # If the user did not supply a grid for lambda, add a column for lambda
    if (auto_lambda) {
        cv_scores$lambda = 0
    }

    # Add a column for the scores
    cv_scores$score = 0

    # Necessary list when lambda is tuned automatically. Keeps track of the
    # selected lambdas for each combination of k and phi
    lambdas_list = list()

    # Compute sample covariance matrix based on the complete data set
    S = stats::cov(X, method = cov_method)

    for (grid_i in 1:nrow(grid)) {
        ## If necessary, begin with computing a sequence for lambda to be used
        ## for the solution path for the given combination of k and phi.
        # Select k and phi
        k = grid$k[grid_i]
        phi = grid$phi[grid_i]

        if (auto_lambda) {
            # Compute weight matrix for the sample covariance matrix based on
            # the complete sample
            W = cggm_weights(S = S, phi = phi, k = k, connected = connected)

            # Initial lambdas
            lambdas = seq(0, 1, 0.1)

            # Compute the solution path, expanding it so that the consecutive
            # solutions for Theta do not differ too much
            res = cggm(S = S, W = W, lambda = lambdas, expand = TRUE)

            # Set lambdas
            lambdas = res$lambdas

            # Store lambdas for later, required when training the final model on
            # the tuned hyperparameters
            lambdas_list[[grid_i]] = lambdas
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
                       expand = FALSE)

            # Compute the cross validation score for each lambda
            for (score_i in 1:length(scores)) {
                scores[score_i] = scores[score_i] - nrow(X_test) / nrow(X) *
                    CGGMR:::.log_likelihood(S_test, get_Theta(res, score_i))
            }
        }

        # If lambda is tuned automatically, select the best performing value to
        # be added to the results. Otherwise, fill in the required values for
        # lambda
        if (auto_lambda) {
            best_index = which.min(scores)
            cv_scores$lambda[grid_i] = lambdas[best_index]
            cv_scores$score[grid_i] = scores[best_index]
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
    res = cggm(S = S, W = W, lambda = lambdas, expand = FALSE)
}
