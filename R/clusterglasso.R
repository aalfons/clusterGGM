#' tag-lasso estimation of the precision matrix
#' @export
#' @description This function computes the tag-lasso estimator for fixed tuning parameters lambda1 and lambda2
#' @param X An (\eqn{n}x\eqn{p})-matrix of \eqn{p} variables and \eqn{n} observations
#' @param pendiag Logical indicator whether or not to penalize the diagonal in Omega. The default is \code{TRUE} (penalization of the diagonal)
#' @param lambda1 Sparsity tuning parameter.
#' @param lambda2 Aggregation tuning parameter.
#' @param rho Starting value for the LA-ADMM tuning parameter. Default is 10^2; will be locally adjusted via LA-ADMM
#' @param it_in Number of inner stages of the LA-ADMM algorithm. Default is 100
#' @param it_out Number of outer stages of the LA-ADMM algorithm. Default is 10
#' @param it_in_refit Number of inner stages of the LA-ADMM algorithm for re-fitting. Default is 100
#' @param it_out_refit Number of outer stages of the LA-ADMM algorithm for re-fitting. Default is 10

#' @return A list with the following components
#' \item{\code{omega_full}}{Estimated (\eqn{p}x\eqn{p}) precision matrix}
#' \item{\code{omega_aggregated}}{Estimated (\eqn{K}x\eqn{K}) precision matrix}
#' \item{\code{cluster}}{Numeric vector indicating the cluster groups for each of the \eqn{p} original variables}
#' \item{\code{M}}{The (\eqn{p}x\eqn{K} membership matrix)}
# clusterglasso <- function(X, pendiag = F,  lambda1, lambda2,
#                             rho = 10^-2, it_in = 100, it_out = 10,  it_in_refit = 100, it_out_refit = 10){
#
#   #### Preliminaries ####
#   # Dimensions
#   p <- ncol(X)
#   n <- nrow(X)
#   S <- stats::cov(X)
#   ominit <- matrix(0, p, p)
#   cinit <- matrix(0, p, p)
#
#   #### Preliminaries for the A matrix ####
#   A_precompute <- preliminaries_for_DOC_subproblem(p = p)
#
#   #### clustergasso ####
#   fit_taglasso <- LA_ADMM_clusterglasso_export(it_out = it_out, it_in = it_in , S = S,
#                                           A =  A, Itilde = A_precompute$Itilde, A_for_C3 = A_precompute$A_for_C3, A_for_T1 = A_precompute$A_for_T1, T2 = A_precompute$T2, T2_for_D = A_precompute$T2_for_D,
#                                           lambda1 = lambda1, lambda2 = lambda2, rho = rho, pendiag = pendiag,
#                                           init_om = ominit, init_u1 = ominit, init_u2 = ominit,
#                                           init_c = cinit, init_u3 = cinit, init_u4 = cinit, init_u5 = cinit)
#
#   #### Level of aggregation and clusters ####
#   # CODE TO BE FILLED OUT, THIS WILL NEED TO BE DETERMINED FROM THE COPY c2
#
#   #### Level of sparsity ####
#   om_P <- (fit_taglasso$c1!=0)*1 # 1 if non-zero
#
#
#   #### Full and Aggregated Precision Matrix  ####
#   # CODE TO BE FILLED OUT, THIS WILL NEED TO BE DETERMINED FROM THE COPY c2
#
#
#
#   out <- list("omega_full" = om_hat_full, "omega_aggregated" = om_hat_agg,
#               "cluster" = cluster, "M" = M)
# }


###############################################################################################################################
##################################################### AUX FUNCTIONS ###########################################################
###############################################################################################################################

preliminaries_for_DOC_subproblem <- function(p){
  # Preliminaries to be used when solving for D, Omega^{(2)} and C^{(3)}; based on A=I_pxp
  # Note that in our case A is always the identity matrix of dimension pxp so we can revise this and make it more efficient


  Itilde <- rbind(diag(1, p), diag(1, p))

  A_for_C3 <- diag(1/2, p)%*%cbind(diag(1,p), diag(1, p))

  A_for_T1 <- diag(1, p+p) - Itilde%*% solve(t(Itilde)%*%Itilde) %*%t(Itilde)
  T2 <- t(cbind(diag(1, p), diag(0,p))) - Itilde%*%solve(t(Itilde)%*%Itilde)%*%diag(1,p)
  T2_for_D <- solve(diag(diag(t(T2)%*%T2)))

  out <- list("Itilde" = Itilde, "A_for_C3" = A_for_C3, "A_for_T1" = A_for_T1, "T2" = T2, "T2_for_D" = T2_for_D)
}
