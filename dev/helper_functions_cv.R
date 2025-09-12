# Some helper functions created for the cross_validation.R file


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
