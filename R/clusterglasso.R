#' tag-lasso estimation of the precision matrix
#' @export
#' @description This function computes the tag-lasso estimator for fixed tuning parameters lambda1 and lambda2
#' @param X An (\eqn{n}x\eqn{p})-matrix of \eqn{p} variables and \eqn{n} observations
#' @param pendiag Logical indicator whether or not to penalize the diagonal in Omega. The default is \code{TRUE} (penalization of the diagonal)
#' @param lambda1 Sparsity tuning parameter.
#' @param lambda2 Aggregation tuning parameter.
#' @param rho Starting value for the LA-ADMM tuning parameter. Default is 10^2; will be locally adjusted via LA-ADMM
#' @param it_in Number of inner stages of the LA-ADMM algorithm. Default is 500
#' @param it_out Number of outer stages of the LA-ADMM algorithm. Default is 10

#' @return A list with the following components
#' \item{\code{omega_full}}{Estimated (\eqn{p}x\eqn{p}) precision matrix}
#' \item{\code{cluster}}{Numeric vector indicating the cluster groups for each of the \eqn{p} original variables}
#' \item{\code{sparsity}}{The (\eqn{p}x\eqn{p} matrix indicating the sparsity pattern in Omega (1=non-zero, 0=zero))}
#' \item{\code{fit}}{Fitted object from LA_ADMM_clusterglasso_export cpp function, for internal use now}
clusterglasso <- function(X, pendiag = F,  lambda1, lambda2, rho = 10^-2, it_in = 500, it_out = 10){

  #### Preliminaries ####
  # Dimensions
  p <- ncol(X)
  n <- nrow(X)
  S <- stats::cov(X)
  ominit <- matrix(0, p, p)
  cinit <- matrix(0, p, p)

  #### Preliminaries for the A matrix ####
  A_precompute <- preliminaries_for_DOC_subproblem(p = p)

  #### clustergasso ####
  fit_taglasso <- LA_ADMM_clusterglasso_export(it_out = it_out, it_in = it_in , S = S,
                                          A =  diag(1,p), Itilde = A_precompute$Itilde, A_for_C3 = A_precompute$A_for_C3, A_for_T1 = A_precompute$A_for_T1, T2 = A_precompute$T2, T2_for_D = A_precompute$T2_for_D,
                                          lambda1 = lambda1, lambda2 = lambda2, rho = rho, pendiag = pendiag,
                                          init_om = ominit, init_u1 = ominit, init_u2 = ominit,
                                          init_c = cinit, init_u3 = cinit, init_u4 = cinit, init_u5 = cinit)

  #### Level of aggregation and clusters ####
  C2 <- fit_taglasso$c2
  digit.acc <- 5 # @Daniel: Should we put this here up until a certain accuracy that they are equal? If so, possible as input argument?
  unique_rows <- unique(round(C2, digit.acc)) # unique rows in C2 are the clusters
  K <- nrow(unique_rows) # Number of clusters = aggregated nodes in the network
  cluster <- rep(NA, K)
  for(i in 1:K){
    check <- apply(C2, 1, unique_rows = unique_rows, function(unique_rows, X){all(round(X, digit.acc)==round(unique_rows[i,], digit.acc))})
    cluster[which(check==T)] <- i
  }
  names(cluster) <- colnames(X)


  #### Level of sparsity ####
  om_P <- (fit_taglasso$c1!=0)*1 # 1 if non-zero
  if(pendiag==F){
    diag(om_P) <- 1 # Diagonal elements are also estimated and are in D
  }




  out <- list("omega_full" = fit_taglasso$om1, "cluster" = cluster, "sparsity" = om_P,
              "fit" = fit_taglasso)
}


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
