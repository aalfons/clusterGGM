#' tag-lasso estimation of the precision matrix
#' @export
#' @description This function computes the tag-lasso estimator for fixed tuning parameters lambda1 and lambda2
#' @param X An (\eqn{n}x\eqn{p})-matrix of \eqn{p} variables and \eqn{n} observations
#' @param A An (\eqn{p}x\eqn{|T|})- binary matrix incorporating the tree-based aggregation structure
#' @param pendiag Logical indicator whether or not to penalize the diagonal in Omega. The default is \code{TRUE} (penalization of the diagonal)
#' @param lambda1 Aggregation tuning parameter. Use \code{taglasso_cv} to let the program determine this tuning parameter based on K-fold cross-validation
#' @param lambda2 Sparsity tuning parameter. Use \code{taglasso_cv} to let the program determine this tuning parameter based on K-fold cross-validation
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
taglasso <- function(X, A, pendiag = F,  lambda1, lambda2,
                     rho = 10^-2, it_in = 100, it_out = 10,  it_in_refit = 100, it_out_refit = 10){

  #### Preliminaries ####
  # Dimensions
  p <- ncol(X)
  n <- nrow(X)
  S <- stats::cov(X)
  ominit <- matrix(0, p, p)
  gaminit <- matrix(0, ncol(A), nrow(A))

  #### Preliminaries for the A matrix ####
  A_precompute <- preliminaries_for_refit_in_R(A = A)

  #### tag-lasso ####
  fit_taglasso <- LA_ADMM_taglasso_export(it_out = it_out, it_in = it_in , S = S,
                                          A =  A, Atilde = A_precompute$Atilde, A_for_gamma = A_precompute$A_for_gamma,
                                          A_for_B = A_precompute$A_for_B, C = A_precompute$C, C_for_D = A_precompute$C_for_D,
                                          lambda1 = lambda1, lambda2 = lambda2,
                                          rho = rho, pendiag = pendiag,
                                          init_om = ominit, init_u1 = ominit, init_u2 = ominit, init_u3 = ominit,
                                          init_gam = gaminit, init_u4 = gaminit, init_u5 = gaminit)

  #### Level of aggregation and clusters ####
  Z <- which(apply(fit_taglasso$gam1, 1, function(U){all(U==0)})==F)
  if(length(Z)==0){
    Z <- nrow(fit_taglasso$gam1)
  }
  AZ <- A[, Z]
  if(length(Z)==1){
    AZ = matrix(AZ, p, 1)
  }
  unique_rows <- unique(AZ)
  K <- nrow(unique_rows) # Number of nodes in aggregated network
  cluster <- rep(NA, K)
  for(i in 1:K){
    check <- apply(AZ, 1, unique_rows = unique_rows, function(unique_rows, X){all(X==unique_rows[i,])})
    cluster[which(check==T)] <- i
  }
  names(cluster) <- colnames(X)

  #### Level of sparsity ####
  om_P <- (fit_taglasso$om3!=0)*1 # 1 if non-zero


  ##### Refit subject to aggregation and sparsity constraints of tag-lasso ####
  gaminit_refit <- matrix(0, ncol(AZ), nrow(AZ))
  prelims_refit <- preliminaries_for_refit_in_R(AZ)
  refit <- refit_LA_ADMM_export(it_out = it_out_refit, it_in = it_in_refit, S = S,
                                A =  AZ,  Atilde = prelims_refit$Atilde, A_for_gamma = prelims_refit$A_for_gamma,
                                A_for_B = prelims_refit$A_for_B, C = prelims_refit$C, C_for_D = prelims_refit$C_for_D,
                                omP = om_P, rho = rho,
                                init_om = ominit, init_u1 = ominit, init_u2 = ominit, init_u3 = ominit,
                                init_gam = gaminit_refit, init_u4 = gaminit_refit, init_u5 = gaminit_refit)


  #### Full and Aggregated Precision Matrix  ####
  omega_block <- AZ%*%refit$gam1
  d <- diag(refit$D)
  re_order <- c()
  M <- matrix(0, p, K) # Membership matrix
  rownames(M) <- colnames(X)
  for(i.c in 1:K){
    re_order <- c(re_order, which(cluster==i.c))
    M[which(cluster==i.c), i.c] = 1
  }
  omega_block_re_order <- omega_block[re_order, re_order]
  omega_block_re_order[lower.tri(omega_block_re_order)] <- t(omega_block_re_order)[lower.tri(omega_block_re_order)]
  colnames(omega_block_re_order) <- rep("", ncol(omega_block_re_order))
  nbr_variables <- 0
  counter <- 1
  # d_agg <- rep(NA, K)
  for(i in 1:K){
    colnames(omega_block_re_order)[1+nbr_variables] <- counter
    nbr_variables <- nbr_variables + length(which(sort(cluster)==i))
    # d_agg[i] <- mean(d[which(sort(cluster)==i)])
    counter = counter + 1
  }
  om_hat_full <- round(omega_block_re_order, 2) + diag(d[re_order])
  rownames(om_hat_full) <- colnames(X)
  colnames(om_hat_full) <- colnames(omega_block_re_order)

  omega_block[lower.tri(omega_block)] <- t(omega_block)[lower.tri(omega_block)]
  omega_aggregated <- MASS::ginv(M)%*%round(omega_block, 4)%*%MASS::ginv(t(M))
  D_agg <- solve(t(M)%*%solve(refit$D)%*%M)
  # om_hat_agg <- omega_aggregated + diag(d_agg)
  om_hat_agg <- omega_aggregated + D_agg
  rownames(om_hat_agg) <- paste0("cluster", 1:K)
  colnames(om_hat_agg) <- 1:K



  out <- list("omega_full" = om_hat_full, "omega_aggregated" = om_hat_agg,
              "cluster" = cluster, "M" = M)
}


###############################################################################################################################
##################################################### AUX FUNCTIONS ###########################################################
###############################################################################################################################

preliminaries_for_refit_in_R <- function(A){
  # Preliminaries on A to be used when solving for D, Omega^{(2)} and Gamma^{(2)}
  z <- ncol(A)
  p <- nrow(A)
  Atilde <- rbind(A, diag(1, z))
  A_for_gamma <- solve(t(A)%*%A + diag(1, z))%*%cbind(t(A), diag(1, z))
  A_for_B <- diag(1, p+z) - Atilde%*% solve(t(Atilde)%*%Atilde) %*%t(Atilde)
  C <- t(cbind(diag(1, p), matrix(0, p, z))) - Atilde%*%solve(t(Atilde)%*%Atilde)%*%t(A)
  C_for_D <- solve(diag(diag(t(C)%*%C)))
  out <- list("A_for_gamma" = A_for_gamma, "Atilde" = Atilde, "A_for_B" = A_for_B, "C" = C, "C_for_D" = C_for_D)
}
