rm(list = ls())

library(mvtnorm)
library(clusterglasso)
library(BayesianOptimization)
source("helper_functions_cv.R")



# Number of observations and variables
n = 100
p = 10


# Generate a precision matrix with a clear block structure
precision = matrix(0, nrow = p, ncol = p)
for (i in 1:p) {
  for (j in 1:p) {
    if ((i <= p/2 && j <= p/2) || (i > p/2 && j > p/2)) {
      precision[i, j] = 1
    }
  }
}
precision = (precision + diag(p))


# Invert it to obtain the covariance matrix used to generate the data
sigma = solve(precision)
set.seed(42)
X = rmvnorm(n, sigma = sigma)


# Get indices of the folds used in the cross validation
folds = create_folds(X, 5, seed = 42)


# Define a function that evaluates the current hyperparameters based on k-fold
# cross validation
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

# Specify the domain for each hyperparameter
domain = matrix(c(0.0, 0.2, 0.0, 0.6), ncol = 2, byrow = TRUE) # Matrix with the domain of each hyperparameter
                                                               # on its rows
                                                               # [0.0, 0.2] <- lambda1 ranges from 0.0 to 0.5
                                                               # [0.0, 0.6] <- lambda2 ranges from 0.0 to 0.6

# Initialize some coarse grid for the first couple of tries of the optimization
# procedure, I choose some points close to the corner points of the domain and
# one in the center
BO_result = matrix(NA, ncol = ncol(domain) + 1, nrow = 5)
BO_result[1, -3] = c(0.05, 0.15)
BO_result[2, -3] = c(0.05, 0.45)
BO_result[3, -3] = c(0.15, 0.15)
BO_result[4, -3] = c(0.15, 0.45)
BO_result[5, -3] = c(0.10, 0.30)

# Get the performance measure for each of the initial points on the grid
for (i in 1:nrow(BO_result)) {
  BO_result[i, 3] = evaluation(X, folds, BO_result[i, -3])
  print(BO_result)
}


# Perform Bayesian Optimization
for (i in 1:10) {
  # Find new set of hyperparameters to try
  res = bayes_opt(BO_result[, -3], BO_result[, 3], domain, kernel_ell = 1.5,
                  kernel_sigma = 1, return_fitted = FALSE)

  # Add a row to BO_result with the new set of hyperparameters and the result
  # from the evaluation
  BO_result = rbind(BO_result, rep(0, ncol(BO_result)))
  BO_result[nrow(BO_result), -ncol(BO_result)] = res$new
  BO_result[nrow(BO_result), ncol(BO_result)] = evaluation(X, folds, res$new[1, ])

  print(BO_result)
}


# Select best performing hyperparameters
lambda1 = BO_result[which.max(BO_result[, 3]), 1]
lambda2 = BO_result[which.max(BO_result[, 3]), 2]

# Compute the weight matrix
S = cov(X)
D = distance(solve(S))
W = exp(-1 * D^2)

# Fit with best set of hyperparameters
fit <- clusterglasso(X, W, lambda1 = lambda1, lambda2 = lambda2)
round(fit$omega_full, 3)
fit$cluster

# Value of the likelihood-based score
lb_score(fit$omega_full, cov(X))

