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
