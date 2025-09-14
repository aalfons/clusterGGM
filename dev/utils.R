# *************************************
# Authors: Andreas Alfons
#          Erasmus University Rotterdam
#
#          Daniel Touw
#          Erasmus University Rotterdam
#
#          Ines Wilms
#          Maastricht University
# *************************************


## Utility functions

# Function to generate grid of lambda values
lambda_grid <- function(bounds, n_points = 10, factor = 1) {
  if (factor == 1) grid <- seq(bounds[1], bounds[2], length.out = n_points)
  else {
    # obtain denominator for base length
    if (factor == 2) denominator <- 2^(n_points-1) - 1
    else denominator <- sum(factor^seq(0, n_points-2))
    # compute base length
    l <- diff(bounds) / denominator
    # initialize grid
    grid <- c(bounds[1], rep.int(NA_real_, n_points-2), bounds[2])
    for (i in seq(2, n_points-1)) {
      grid[i] <- grid[i-1] + factor^(i-2)*l
    }
  }
  # return grid of values
  grid
}

# Function to set up folds for K-fold cross-validation
cv_folds <- function(n, K) {
  # permute observations
  indices <- sample.int(n)
  # assign a block to each observation
  blocks <- rep(seq_len(K), length.out = n)
  # split the permuted observations according to the block they belong to
  folds <- split(indices, blocks)
  names(folds) <- NULL
  folds
}

# Function to compute distances based on "valid" comparisons
distance <- function(Omega) {
  # extract number of variables
  p <- ncol(Omega)
  seq_p <- seq_len(p)
  # compute distances based on "valid" comparisons
  sapply(seq_len(p), function(j) {
    sapply(seq_len(p), function(i) {
      sum((Omega[i, -c(i, j)] - Omega[j, -c(i, j)])^2)^0.5
    })
  })
}

# Function to compute the likelihood-based score from Wilms & Bien (2022), I
# take the negative of the measure in the paper, as the Bayesian optimization
# code is written for maximization
lb_score <- function(Omega, S) {
  log(det(Omega)) - sum(diag(S %*% Omega))
}


## Functions to evaluate methods for specific values of the hyperparameters

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

    # compute the distances based on "valid" comparisons between the
    # rows of the precision matrix
    D <- distance(solve(S_estimation))

    # fit method on the estimation sample
    fit <- taglasso(X_estimation, A, lambda1 = lambda1, lambda2 = lambda2, ...)

    # compute score on hold-out sample
    score <- score + lb_score(fit$omega_full, S_holdout)

  }

  # return average score over the hold-out samples
  score / K

}


## Functions to select the optimal tuning parameters

# Function to select the optimal tuning parameters for taglasso
cv_taglasso <- function(X, A, lambda1, lambda2, folds,
                        type = c("bayes", "grid"),
                        init = NULL, it_bayes = 10, ...) {

  ## initializations
  type <- match.arg(type)

  if (type == "bayes") {

    ## Bayesian optimization

    # Matrix with the domain of each hyperparameter
    domain <- rbind(lambda1, lambda2)

    # Initialize some coarse grid for the first couple of tries of the Baysian
    # optimization procedure, I choose some points close to the corner points
    # of the domain and one in the center
    if (is.null(init)) {
      grid <- cbind(
        lambda1[1] + c(0.25, 0.25, 0.75, 0.75, 0.5) * diff(lambda1),
        lambda2[1] + c(0.25, 0.75, 0.25, 0.75, 0.5) * diff(lambda2)
      )
    } else grid <- init

    # Evaluate the method on the initial grid
    scores <- mapply(function(lambda1, lambda2, ...) {
      evaluate_taglasso(X, A, lambda1 = lambda1, lambda2 = lambda2,
                        folds = folds, ...)
    }, lambda1 = grid[, 1], lambda2 = grid[, 2], ...)

    # Perform Bayesian Optimization
    for (i in seq_len(it_bayes)) {
      # Find a new set of hyperparameters to try
      new_lambda <- bayes_opt(grid, scores, domain, kernel_ell = 1.5,
                              kernel_sigma = 1, return_fitted = FALSE)$new
      new_score <- evaluate_taglasso(X, A, lambda1 = new_lambda[1, 1],
                                     lambda2 = new_lambda[1, 2],
                                     folds = folds, ...)

      # Add a row to grid with the new set of hyperparameters
      grid <- rbind(grid, new_lambda)
      # Add the result from the evaluation to scores
      scores <- c(scores, new_score)
    }

  } else {

    ## cross-validation over a grid of parameter values

    # define grid of hyperparameters
    grid <- expand.grid(lambda1 = lambda1, lambda2 = lambda2)

    # compute scores
    scores <- mapply(function(lambda1, lambda2, ...) {
      evaluate_taglasso(X, A, lambda1 = lambda1, lambda2 = lambda2,
                        folds = folds, ...)
    }, lambda1 = grid$lambda1, lambda2 = grid$lambda2, ...)

  }

  # Select best performing hyperparameters
  select <- which.max(scores)
  lambda1_opt <- grid[select, 1]
  lambda2_opt <- grid[select, 2]

  # Return list of best performing hyperparameters
  list(lambda1_opt = lambda1_opt, lambda2_opt = lambda2_opt,
       score_opt = scores[select])

}


## find smallest aggregation parameter that aggregates everything into one
## cluster for clusterglasso() and taglasso()

# binary search for smallest lambda2 that aggregates all variables into one
# cluster (or aggregates as much as possible if knn-weights don't yield a
# connected graph)
binary_search <- function(target = "lambda", lower = 0, upper = 10,
                          tol = 0.05, max_it = 50, fun = taglasso,
                          args = list(...), ...) {

  ## initializations
  length <- upper - lower
  lst <- vector(mode = "list", length = 1)
  names(lst) <- target

  ## fit with initial bounds
  lst[[target]] <- lower
  fit_lower <- do.call(fun, c(lst, args))
  lst[[target]] <- upper
  fit_upper <- do.call(fun, c(lst, args))

  ## determine the number of clusters
  # nr_lower <- nr_clusters(fit_lower)
  # if (nr_lower == 1) stop("choose a smaller value for 'lower'")
  # nr_upper <- nr_clusters(fit_upper)
  # if (nr_upper > 1) stop("choose a larger value for 'upper'")
  nr_lower <- nr_clusters(fit_lower)
  nr_upper <- nr_clusters(fit_upper)
  if (nr_lower <= nr_upper) {
    stop("choose a smaller value for 'lower' or a larger value for 'upper'")
  }

  ## binary search
  it <- 0
  while (length >= tol && it < max_it) {
    # fit method with regularization parameter halfway in between
    tmp <- (upper + lower) / 2
    lst[[target]] <- tmp
    fit_tmp <- do.call(fun, c(lst, args))
    nr_tmp <- nr_clusters(fit_tmp)
    # update lower or upper bound depending on the resulting number of clusters
    if (nr_tmp == nr_upper) {
      upper <- tmp
      nr_upper <- nr_tmp
    } else {
      lower <- tmp
      nr_lower <- nr_tmp
    }
    # update length of interval
    length <- upper - lower
    it <- it + 1
  }

  ## return smallest value that aggregates everything as much as possible
  upper
}

# function to extract number of clusters
nr_clusters <- function(fit) length(unique(fit$cluster))


## functions to construct the A matrix for the tag-lasso

# required packages
library("rare")

# function to obtain ideal, realistic, and misspecified tree aggregation matrix
# blocks ......... integer vector indicating to which block each variable
#                  belongs
# tau ............ scaling factor for the variance of the normal distribution
#                  from which the latent points are drawn when constructing the
#                  realistic tree aggregation matrix
# misspecified ... character string specifying whether the misspecified tree
#                  is obtained by flipping block membership for some variables
#                  or by permuting block membership
# prob_flip ...... probability for flipping block membership of each variable
#                  (if misspecified = "flip")
get_tree_matrices <- function(blocks, tau = 0.05,
                              misspecified = c("flip", "permutation"),
                              prob_flip = 0.1) {

  ## initializations
  misspecified <- match.arg(misspecified)

  ## obtain number of variables and blocks
  p <- length(blocks)      # overall number of variables
  p_b <- tabulate(blocks)  # number of variables per block
  B <- length(p_b)         # number of blocks

  ## ideal tree aggregation matrix
  A_ideal <- cbind(
    diag(p),                                                                      # leaves
    if (B > 1 && B < p) sapply(seq_len(B), function(b) as.integer(blocks == b)),  # second level
    rep.int(1, p)                                                                 # root
  )

  ## realistic tree aggregation matrix
  # define centroids
  centroids <- 1 / seq(from = B, to = 1)
  # compute pairwise differences
  if (B > 1) {
    # for each centroid, find the smallest absolute distance to another centroid
    dist_centroids_min <- sapply(seq_len(B),
                                 function(i) min(abs(centroids[i] - centroids[-i])))
  } else dist_centroids_min <- 1
  # define error terms
  errors <- rnorm(p)
  # draw latent points from a normal distribution
  Z_realistic <- centroids[blocks] + tau * dist_centroids_min[blocks] * errors
  # apply hierarchical clustering to latent points
  hc_realistic <- hclust(dist(Z_realistic))
  # convert tree from hierarchical clustering to tree aggregation matrix
  A_realistic <- as.matrix(rare::tree.matrix(hc_realistic))

  ## misspecified tree aggregation matrix
  if (B == 1 || B == p) {
    # not meaningful for the unstructured design, as the true clustering is
    # always part of the tree (all variables belong to one cluster, or each
    # variable is its own cluster)
    A_misspecified <- NULL
  } else {
    # flip block membership for certain variables or permute block membership
    if (misspecified == "flip") {
      # with certain probability, flip each variable to be in the next block
      # (variables in the last block are flipped to be in the first block)
      probabilities <- runif(p)
      blocks_flipped <- ifelse(blocks == B, 1, blocks+1)
      blocks_misspecified <- ifelse(probabilities < prob_flip,
                                    blocks_flipped, blocks)
    } else {
      permutation <- sample.int(p)
      blocks_misspecified <- blocks[permutation]
    }
    # draw latent points from a normal distribution
    Z_misspecified <- centroids[blocks_misspecified] +
      tau * dist_centroids_min[blocks_misspecified] * errors
    # apply hierarchical clustering to latent points
    hc_misspecified <- hclust(dist(Z_misspecified))
    # convert tree from hierarchical clustering to tree aggregation matrix
    A_misspecified <- as.matrix(rare::tree.matrix(hc_misspecified))
  }

  # return ideal and realistic matrices
  list(ideal = A_ideal, realistic = A_realistic, misspecified = A_misspecified)
}
