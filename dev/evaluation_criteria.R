# ************************************
# Author: Andreas Alfons
#         Erasmus University Rotterdam
# ************************************


## evaluation criteria used by Pircalabelu & Claeskens (2020)

# Frobenius norm (root mean squared error)
Frobenius <- function(true, estimate) norm(true - estimate, type = "F")

# Negative log likelihood
neg_log_lik <- function(Sigma, estimate) {
  -log(det(estimate)) + sum(diag(estimate %*% Sigma))
}

# Rand index
Rand_index <- function(true, estimate) {
  # number of individuals
  n <- length(true)
  # determine which individuals are in the same communities
  true_mat <- sapply(true, function(x) as.integer(x == true))
  estimate_mat <- sapply(estimate, function(x) as.integer(x == estimate))
  # count disagreements between communities, i.e., pairs that have different
  # relationship in the two communities:
  #  - Each point has the same relationship with itself in the two communities.
  #    The diagonal therefore does not contribute, we don't have to exclude it
  #    explicitly in the calculation.
  #  - Each pair of different individuals is counted twice (lower and upper
  #    triangle).
  disagreements <- sum(abs(true_mat - estimate_mat)) / 2
  # compute Rand index
  1 - disagreements / choose(n, 2)
}

# adjusted Rand index (corrected-for-chance version of the Rand index)
adjusted_Rand_index <- function(true, estimate) {
  # calculation below does not work if each item is its own cluster
  # seq_p <- seq_along(true)
  # if (identical(unname(true), seq_p) && identical(unname(estimate), seq_p)) {
  #   return(1.0)
  # }
  # the calculation also doesn't work if all items form one cluster
  if (identical(unname(true), unname(estimate))) return(1.0)
  # compute contingency table
  n_ij <- table(true, estimate)
  a_i <- rowSums(n_ij)
  b_j <- colSums(n_ij)
  n <- sum(n_ij)
  # compute sums of binomial coefficients
  sum_n_ij_2 <- sum(choose(n_ij, 2))
  sum_a_i_2 <- sum(choose(a_i, 2))
  sum_b_j_2 <- sum(choose(b_j, 2))
  n_2 <- choose(n, 2)
  # compute adjusted Rand index
  tmp <- sum_a_i_2 * sum_b_j_2 / n_2
  (sum_n_ij_2 - tmp) / (0.5 * (sum_a_i_2 + sum_b_j_2) - tmp)
}


## other evaluation criteria

# F_beta score: it is assumed that input matrices have values are 0/1
F_score <- function(true, estimate, beta = 1) {
  confusion_matrix <- table(true, estimate)
  wTP <- (1 + beta^2) * confusion_matrix[2, 2]
  wTP / (wTP + beta^2 * confusion_matrix[2, 1] + confusion_matrix[1, 2])
}

# false positive rate: true negatives are assumed to be exactly zero, but
# estimated values allow for some very small numerical tolerance
FPR <- function(true, estimate, tol = .Machine$double.eps^0.5) {
  # indices for upper triangle
  is_upper <- upper.tri(true, diag = FALSE)
  # find true negatives
  is_TN <- true[is_upper] == 0
  TN <- sum(is_TN)
  if (TN == 0) return(NA_real_)
  # which of those are estimated positives?
  FP <- sum(abs(estimate[is_upper][is_TN]) >= tol)
  # return false positive rate
  FP / TN
}

# false negative rate: true positives are assumed to be exactly nonzero, but
# estimated values allow for some very small numerical tolerance
FNR <- function(true, estimate, tol = .Machine$double.eps^0.5) {
  # indices for upper triangle
  is_upper <- upper.tri(true, diag = FALSE)
  # find true positives
  is_TP <- true[is_upper] != 0
  TP <- sum(is_TP)
  if (TP == 0) return(NA_real_)
  # which of those are estimated negatives?
  FN <- sum(abs(estimate[is_upper][is_TP]) < tol)
  # return false positive rate
  FN / TP
}
