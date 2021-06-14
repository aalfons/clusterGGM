#' tuning parameter selection for cluster-glasso with Bayesian Optimization
#' @export
#' @description This function selects the tuning parameter for the cluster-glasso with Bayesian Optimization
#' @param X An (\eqn{n}x\eqn{p})-matrix of \eqn{p} variables and \eqn{n} observations
#' @param lambda1_lb Lower bound in the domain of the sparsity tuning parameter
#' @param lambda1_ub Upper bound in the domain of the sparsity tuning parameter
#' @param lambda2_lb Lower bound in the domain of the aggregation tuning parameter
#' @param lambda2_ub Upper bound in the domain of the aggregation tuning parameter
#' @param bo_init Initialization matrix for Bayesian optimization. We advise to let the program determine the initialization matrix based on the provided lower and upper bounds of the tuning parameters
#' @param fold Folds to be used in cross-validation procedure. Default is 5-fold cross-validation
#' @param seed_cv Seed for creating folds in cross-validation procedure
#' @param bo_iter Number of iterations for Bayesian optimization.
#' #' @return A list with the following components
#' \item{\code{lambda1_opt}}{Selected sparsity tuning parameter}
#' \item{\code{lambda2_opt}}{Selected aggregation tuning parameter}
tuning_clusterglasso <- function(X, lambda1_lb = 0, lambda1_ub = 0.2, lambda2_lb = 0, lambda2_ub = 0.6,
                             bo_init = NULL, fold = 5, seed_cv, bo_iter = 10){

  # Specify the domain for each hyperparameter
  domain = matrix(c(lambda1_lb, lambda1_ub, lambda2_lb, lambda2_ub), ncol = 2, byrow = TRUE) # Matrix with the domain of each hyperparameter

  # Initialize some coarse grid for the first couple of tries of the Baysian optimization
  # procedure, I choose some points close to the corner points of the domain and
  # one in the center
  if(is.null(bo_init)){
    bo_init = matrix(NA, ncol = ncol(domain) + 1, nrow = 5)
    bo_init[1, -3] = c(0.25*lambda1_ub, 0.25*lambda2_ub)
    bo_init[2, -3] = c(0.25*lambda1_ub, 0.75*lambda2_ub)
    bo_init[3, -3] = c(0.75*lambda1_ub, 0.25*lambda2_ub)
    bo_init[4, -3] = c(0.75*lambda1_ub, 0.75*lambda2_ub)
    bo_init[5, -3] = c(0.5*lambda1_ub, 0.5*lambda2_ub)
  }

  folds = create_folds(X, fold, seed = seed_cv)

  for (i in 1:nrow(bo_init)) {
    bo_init[i, 3] = evaluation(X, folds, bo_init[i, -3]) # IW: note we can also pass on the additional arguments of clusterglasso if needed
  }

  # Perform Bayesian Optimization
  for (i in 1:bo_iter) {
    # Find new set of hyperparameters to try
    res = bayes_opt(bo_init[, -3], bo_init[, 3], domain, kernel_ell = 1.5,
                    kernel_sigma = 1, return_fitted = FALSE, seed = i) # <- important that seed changes each iteration

    # Add a row to bo_init with the new set of hyperparameters and the result
    # from the evaluation
    bo_init = rbind(bo_init, rep(0, ncol(bo_init)))
    bo_init[nrow(bo_init), -ncol(bo_init)] = res$new
    bo_init[nrow(bo_init), ncol(bo_init)] = evaluation(X, folds, res$new[1, ])
  }

  # Select best performing hyperparameters
  lambda1 = bo_init[which.max(bo_init[, 3]), 1]
  lambda2 = bo_init[which.max(bo_init[, 3]), 2]

  out <- list("lambda1_opt" = lambda1, "lambda2_opt" = lambda2)


}


#### HELPER FUNCTIONS ####
evaluation <- function(X, folds, hyper_params)
{
  # Set hyperparameters
  lambda1 = hyper_params[1]
  lambda2 = hyper_params[2]
  phi = 1

  # Initialize score
  score = 0

  for (i in 1:length(folds)) {
    # Divide X into the estimation and hold-out samples
    X_noFk = X[-folds[[i]], ]
    X_Fk = X[folds[[i]], ]

    # Compute in- and out-of-sample covariance matrices
    S_noFk = cov(X_noFk)
    S_Fk = cov(X_Fk)

    # Compute the distances based on "valid" comparisons between the rows of the
    # precision matrix
    D = distance(solve(S_noFk))

    # Compute W
    W = exp(-phi * D^2)

    # Minimize the loss
    fit <- clusterglasso(X_noFk, W, lambda1 = lambda1, lambda2 = lambda2)

    # Add the score
    score = score + lb_score(fit$omega_full, S_Fk)
  }

  return(score / length(folds))
}

# Function to compute distances based on "valid" comparisons
distance <- function(omega)
{
  k = ncol(omega)
  result = matrix(0, ncol = k, nrow = k)

  for (i in 1:k) {
    for (j in 1:k) {
      result[i, j] = sum((omega[i, -c(i, j)] - omega[j, -c(i, j)])^2)^0.5
    }
  }

  return(result)
}


# Function to compute the likelihood-based score from Wilms & Bien (2021), I
# take the negative of the measure in the paper, as the Bayesian optimization
# code is written for maximization
lb_score <- function(Omega, S)
{
  return(log(det(Omega)) - sum(diag(S %*% Omega)))
}


# Function to create folds for k-fold cross validation
create_folds <- function(X, k, seed = NA)
{
  n = nrow(X)
  n_i = round(n/k)

  if (!is.na(seed)) {
    set.seed(seed)
  }
  indices = sample(1:n)

  folds = list()
  for (i in 1:k) {
    folds[[i]] = indices[((i - 1) * n_i + 1):(min(i * n_i, n))]
    names(folds)[i] = paste("f", i, sep = "")
  }

  return(folds)
}

