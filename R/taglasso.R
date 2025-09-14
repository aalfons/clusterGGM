#' Tree-Aggregated Graphical Lasso
#'
#' Compute the tree-aggregated graphical lasso (tag-lasso) estimator of the precision matrix for fixed values of the tuning parameters.
#'
#' @param X An (\eqn{n}x\eqn{p})-matrix of \eqn{p} variables and \eqn{n} observations
#' @param A An (\eqn{p}x\eqn{|T|})- binary matrix incorporating the tree-based aggregation structure
#' @param pendiag Logical indicator whether or not to penalize the diagonal in Omega. The default is \code{FALSE} (no penalization of the diagonal)
#' @param lambda1 Aggregation tuning parameter. Use \code{taglasso_cv} to let the program determine this tuning parameter based on K-fold cross-validation
#' @param lambda2 Sparsity tuning parameter. Use \code{taglasso_cv} to let the program determine this tuning parameter based on K-fold cross-validation
#' @param adaptive Logical indicator whether an adaptive lasso type of sparsity penalty should be used or not. Default is FALSE
#' @param power_adaptive Power for the weights int he adaptive lasso type of sparsity penalty. Default is one.
#' @param W_sparsity An (\eqn{p}x\eqn{p})-matrix of weights for the sparsity penalty term. Only relevant if adaptive = TRUE
#' @param rho Starting value for the LA-ADMM tuning parameter. Default is 10^2; will be locally adjusted via LA-ADMM
#' @param it_in Number of inner stages of the LA-ADMM algorithm. Default is 100
#' @param it_out Number of outer stages of the LA-ADMM algorithm. Default is 10
#' @param refitting Logical indicator whether refitting subject to sparsity and aggregation constraints is done or not. The default is \code{TRUE}.
#' @param it_in_refit Number of inner stages of the LA-ADMM algorithm for re-fitting (Only relevant if \code{refitting = TRUE}). Default is 100
#' @param it_out_refit Number of outer stages of the LA-ADMM algorithm for re-fitting (Only relevant if \code{refitting = TRUE}). Default is 10
#'
#' @return A list with the following components
#' \item{\code{omega_full}}{Estimated (\eqn{p}x\eqn{p}) precision matrix}
#' \item{\code{cluster}}{Numeric vector indicating the cluster groups for each of the \eqn{p} original variables}
#' \item{\code{sparsity}}{The (\eqn{p}x\eqn{p} matrix indicating the sparsity pattern in Omega (1=non-zero, 0=zero))}
#' \item{\code{fit}}{Fitted object from LA_ADMM_taglasso_export cpp function, for internal use now}
#' \item{\code{refit}}{Fitted object from refit_LA_ADMM_export cpp function, for internal use now}
#'
#' @author Ines Wilms and Jacob Bien
#'
#' @references
#' I. Wilms and J. Bien (2022) Tree-based Node Aggregation in Sparse Graphical
#' Models. \emph{Journal of Machine Learning Research}, \bold{23}(243), 1--36.
#' https://jmlr.org/papers/v23/21-0105.html
#'
#' @useDynLib clusterGGM
#' @export
taglasso <- function(X, A, pendiag = F,  lambda1, lambda2, adaptive = FALSE, power_adaptive = 1, W_sparsity = NULL,
                     rho = 10^-2, it_in = 100, it_out = 10, refitting = T,  it_in_refit = 100, it_out_refit = 10){

  #### Preliminaries ####
  # Dimensions
  p <- ncol(X)
  n <- nrow(X)
  S <- stats::cov(X)
  ominit <- matrix(0, p, p)
  gaminit <- matrix(0, ncol(A), nrow(A))

  if(is.null(W_sparsity) & adaptive == TRUE){
    # Compute the weight matrix
    W_sparsity = base::abs(1/solve(S))^power_adaptive
  }

  if(adaptive==FALSE){ # no weights if standard lasso type of penalty
    W_sparsity = matrix(1, p, p)
  }

  #### Preliminaries for the A matrix ####
  A_precompute <- preliminaries_for_refit_in_R(A = A)

  #### tag-lasso ####
  fit_taglasso <- LA_ADMM_taglasso_export(it_out = it_out, it_in = it_in , S = S, W_sparsity = W_sparsity,
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
  gam1_red <- fit_taglasso$gam1[Z, ]
  if(length(Z)==1){
    AZ = matrix(AZ, p, 1)
    gam1_red = matrix(gam1_red, 1, p)
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
  if(pendiag==F){
    diag(om_P) <- 1 # Diagonal elements are also estimated and are in D
  }

  if(refitting){
    ##### Refit subject to aggregation and sparsity constraints of tag-lasso ####
    gaminit_refit <- matrix(0, ncol(AZ), nrow(AZ))
    prelims_refit <- preliminaries_for_refit_in_R(AZ)
    refit <- refit_LA_ADMM_export(it_out = it_out_refit, it_in = it_in_refit, S = S,
                                  A =  AZ,  Atilde = prelims_refit$Atilde, A_for_gamma = prelims_refit$A_for_gamma,
                                  A_for_B = prelims_refit$A_for_B, C = prelims_refit$C, C_for_D = prelims_refit$C_for_D,
                                  omP = om_P, rho = rho,
                                  init_om = ominit, init_u1 = ominit, init_u2 = ominit, init_u3 = ominit,
                                  init_gam = gaminit_refit, init_u4 = gaminit_refit, init_u5 = gaminit_refit)

    omega_full <- refit$om1

    # CODE TO RE-ORDER OMEGA TO BETTER VISUALIZE the CLUSTERS
    #### Full and Aggregated Precision Matrix  ####
    # omega_block <- AZ%*%refit$gam1 # gam 1 : Groupwise soft-thresholding
    # rownames(omega_block) <- colnames(omega_block) <- paste0("p", 1:p)
    # d <- diag(refit$D)
    # re_order <- c()
    # M <- matrix(0, p, K) # Membership matrix
    # rownames(M) <- colnames(X)
    # for(i.c in 1:K){
    #   re_order <- c(re_order, which(cluster==i.c))
    #   M[which(cluster==i.c), i.c] = 1
    # }

    # omega_block_re_order <- omega_block[re_order, re_order]
    # omega_block_re_order[lower.tri(omega_block_re_order)] <- t(omega_block_re_order)[lower.tri(omega_block_re_order)]
    # colnames(omega_block_re_order) <- rep("", ncol(omega_block_re_order))
    # nbr_variables <- 0
    # counter <- 1
    # # d_agg <- rep(NA, K)
    # for(i in 1:K){
    #   colnames(omega_block_re_order)[1+nbr_variables] <- counter
    #   nbr_variables <- nbr_variables + length(which(sort(cluster)==i))
    #   # d_agg[i] <- mean(d[which(sort(cluster)==i)])
    #   counter = counter + 1
    # }
    # om_hat_full <- round(omega_block_re_order, 2) + diag(d[re_order])
    # rownames(om_hat_full) <- colnames(X)
    # colnames(om_hat_full) <- colnames(omega_block_re_order)

  }else{
    omega_full <- fit_taglasso$om1
    refit <- NULL
  }

  out <- list("omega_full" = omega_full, "cluster" = cluster, "sparsity" = om_P, "fit" = fit_taglasso,
              "refit" = refit)
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
