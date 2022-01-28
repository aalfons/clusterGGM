#' cluster-glasso estimation of the precision matrix with permutation matrix such that diagonal elements are taken into account
#' @export
#' @description This function computes the cluster-lasso estimator (thereby also accounting for the diagonal elements when clustering) for fixed tuning parameters lambda1 and lambda2
#' @param X An (\eqn{n}x\eqn{p})-matrix of \eqn{p} variables and \eqn{n} observations
#' @param W An (\eqn{p}x\eqn{p})-matrix of weights
#' @param pendiag Logical indicator whether or not to penalize the diagonal in Omega. The default is \code{FALSE} (no penalization of the diagonal)
#' @param lambda1 Sparsity tuning parameter.
#' @param adaptive Logical indicator whether an adaptive lasso type of sparsity penalty should be used or not. Default is FALSE
#' @param power_adaptive Power for the weights int he adaptive lasso type of sparsity penalty. Default is one.
#' @param W_sparsity An (\eqn{p}x\eqn{p})-matrix of weights for the sparsity penalty term. Only relevant if adaptive = TRUE
#' @param lambda2 Aggregation tuning parameter.
#' @param knn_weights Boolean to indicate whether knn weights are used.
#' @param phi A scalar to tune the weights in the W for clustering
#' @param knn The number of nearest neighbors used for the W for clustering.
#' @param eps_fusions Threshold for fusing clusters. Default is 10^-3
#' @param rho Starting value for the LA-ADMM tuning parameter. Default is 10^2; will be locally adjusted via LA-ADMM
#' @param it_in Number of inner stages of the LA-ADMM algorithm. Default is 100
#' @param it_out Number of outer stages of the LA-ADMM algorithm. Default is 10
#' @param refitting Logical indicator whether refitting subject to sparsity and aggregation constraints is done or not. The default is \code{TRUE}.
#' @param it_in_refit Number of inner stages of the LA-ADMM algorithm for re-fitting (Only relevant if \code{refitting = TRUE}). Default is 100
#' @param it_out_refit Number of outer stages of the LA-ADMM algorithm for re-fitting (Only relevant if \code{refitting = TRUE}). Default is 10

#' @return A list with the following components
#' \item{\code{omega_full}}{Estimated (\eqn{p}x\eqn{p}) precision matrix}
#' \item{\code{cluster}}{Numeric vector indicating the cluster groups for each of the \eqn{p} original variables}
#' \item{\code{sparsity}}{The (\eqn{p}x\eqn{p} matrix indicating the sparsity pattern in Omega (1=non-zero, 0=zero))}
#' \item{\code{fit}}{Fitted object from LA_ADMM_clusterglasso_export cpp function, for internal use now}
#' \item{\code{refit}}{Fitted object from refit_LA_ADMM_export cpp function, for internal use now}
clusterglasso_permutation <- function(X, W = NULL, pendiag = F,  lambda1, adaptive = FALSE, power_adaptive = 1, W_sparsity = NULL, lambda2,
                                      knn_weights = F, knn_connect = F, phi = 1, knn = 3, eps_fusions = 1e-3, rho = 10^-2, it_in = 100,
                                      it_out = 10, refitting = T,  it_in_refit = 100, it_out_refit = 10) {

  #### Preliminaries ####
  # Dimensions
  p <- ncol(X)
  n <- nrow(X)
  S <- stats::cov(X)
  ominit <- matrix(0, p, p)

  if(is.null(W)){ #IW: I've put our default for weight matrix here
    # Compute the weight matrix
    D = distance(solve(S))
    W = exp(-phi * D^2)

    if (knn_weights) {
      W = .knn_weights(D, W, knn, knn_connect)
    }
  }

  if(is.null(W_sparsity) & adaptive==TRUE){
    # Compute the weight matrix
    W_sparsity = base::abs(1/solve(S))^power_adaptive
  }

  if(adaptive==FALSE){ # no weights if standard lasso type of penalty
    W_sparsity = matrix(1, p, p)
  }

  #### clustergasso ####
  fit_clusterglasso <- LA_ADMM_clusterglasso_permutation_export(it_out = it_out, it_in = it_in , S = S, W = W, W_sparsity = W_sparsity,
                                                                lambda1 = lambda1, lambda2 = lambda2, eps_fusions = eps_fusions, rho = rho,
                                                                pendiag = pendiag, init_om = ominit, init_u1 = ominit, init_u2 = ominit,
                                                                init_u3 = ominit)

  #### Level of aggregation and clusters ####
  C2 <- fit_clusterglasso$om2
  unique_rows <- unique(round(C2, nchar(eps_fusions)-2)) #IW: round in line with eps_fusions argument
  K <- nrow(unique_rows) # Number of clusters = aggregated nodes in the network
  cluster <- rep(NA, K)
  for(i in 1:K){
    check <- apply(C2, 1, unique_rows = unique_rows, function(unique_rows, X){all(round(X, nchar(eps_fusions)-2)==round(unique_rows[i,], nchar(eps_fusions)-2))})
    cluster[which(check==T)] <- i
  }
  names(cluster) <- colnames(X)


  #### Level of sparsity ####
  om_P <- (fit_clusterglasso$om1!=0)*1 # 1 if non-zero
  if(pendiag==F){
    diag(om_P) <- 1 # Diagonal elements are also estimated and are in D
  }


  if(refitting){
    ##### A matrix that encodes aggregation ####
    AZ <- matrix(0, p, max(cluster)+1)
    AZ[, ncol(AZ)] <- 1
    for(i in 1:max(cluster)){
      members <- which(cluster==i)
      AZ[members, i] <- 1
    }

    ##### Refit subject to aggregation and sparsity constraints of tag-lasso ####
    gaminit_refit <- matrix(0, ncol(AZ), nrow(AZ))
    prelims_refit <- preliminaries_for_refit_in_R(AZ)
    refit <- refit_LA_ADMM_export(it_out = it_out_refit, it_in = it_in_refit, S = S,
                                  A =  AZ,  Atilde = prelims_refit$Atilde, A_for_gamma = prelims_refit$A_for_gamma,
                                  A_for_B = prelims_refit$A_for_B, C = prelims_refit$C, C_for_D = prelims_refit$C_for_D,
                                  omP = om_P, rho = rho,
                                  init_om = ominit, init_u1 = ominit, init_u2 = ominit, init_u3 = ominit,
                                  init_gam = gaminit_refit, init_u4 = gaminit_refit, init_u5 = gaminit_refit)

    om_hat_full <- refit$om1

  }else{
    om_hat_full <- fit_taglasso$om1
    refit <-  NULL
  }


  out <- list("omega_full" = om_hat_full, "cluster" = cluster, "sparsity" = om_P, "fit" = fit_clusterglasso,
              "refit" = refit, "W_aggregation" = W)
}
