#include <RcppArmadillo.h>
#include "convex_clustering.h"
#include "fusions.h"


double cluster_loss(const arma::mat& c2, const arma::mat& X, const arma::mat& W, const double& rho, const double& lambda2)
{
  // Input
  // c2 : matrix of dimensions p times p which we use to minimize the loss
  // X : matrix of dimensions p times p which is the "anchor point" in the minimization. In our case this X = cold - u5 / rho
  // W : matrix of dimensions p times p which contains the weights used to reflect importance of clustering i and j
  // rho : scalar, parameter ADMM
  // lambda2 : scalar, regularization parameter clustering

  // Function : Compute the convex clustering loss for a given c2

  // Output
  // result : scalar, loss

  // Preliminary
  int p = c2.n_rows;

  // Compute first term of the loss function and initialize the value for the penalty as zero
  double term1 = rho / 2 * pow(arma::norm(c2 - X, "fro"), 2.0);
  double penalty = 0.0;

  // Compute the penalty
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < i; j++) {
      penalty += W(i, j) * arma::norm(c2.row(i) - c2.row(j));
    }
  }

  // Add both terms
  double result = term1 + lambda2 * penalty;

  return result;
}


double cluster_loss_fusions(const arma::mat& M, const arma::mat& X, const arma::sp_mat U, const arma::mat& UWU, const double& rho, const double& lambda2)
{
  // Input
  // M : centroid matrix of dimensions c times p which we use to minimize the loss
  // X : matrix of dimensions p times p which is the "anchor point" in the minimization. In our case this X = cold - u5 / rho
  // UWU : matrix of dimensions c times c which contains the weights used to reflect importance of clustering i and j
  // rho : scalar, parameter ADMM
  // lambda2 : scalar, regularization parameter clustering

  // Function : Compute the convex clustering loss for a given c2

  // Output
  // result : scalar, loss

  // Preliminary
  int c = M.n_rows;

  // Compute first term of the loss function and initialize the value for the penalty as zero
  double term1 = rho / 2 * pow(arma::norm(X - U * M, "fro"), 2.0);
  double penalty = 0.0;

  // Compute the penalty
  for (int i = 0; i < c; i++) {
    for (int j = 0; j < i; j++) {
      if (UWU(i, j) > 0) {
        penalty += UWU(i, j) * arma::norm(M.row(i) - M.row(j));
      }
    }
  }

  // Add both terms
  double result = term1 + lambda2 * penalty;

  return result;
}


void CMM1_update(arma::mat& c2, const arma::mat& X, const arma::mat& W, const double& rho, const double& lambda2)
{
  // Input
  // c2 : matrix of dimensions p times p which we use to minimize the loss
  // X : matrix of dimensions p times p which is the "anchor point" in the minimization. In our case this X = cold - u5 / rho
  // W : matrix of dimensions p times p which contains the weights used to reflect importance of clustering i and j
  // rho : scalar, parameter ADMM
  // lambda2 : scalar, regularization parameter clustering

  // Function : Computes the update for c2

  // Output
  // void, c2 is passed by reference and adjusted inside the function

  // Preliminaries
  int p = c2.n_rows;
  arma::mat V(p, p, arma::fill::zeros);

  // Fill the matrix V
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < i; j++) {
      double temp = W(i, j) / std::max(arma::norm(c2.row(i) - c2.row(j)), 1e-6);

      V(i, i) += temp;
      V(j, j) += temp;
      V(i, j) -= temp;
      V(j, i) -= temp;
    }
  }

  // Compute the update as c2 = (rho * I_pxp + lambda2 * V)^-1 * rho * X
  V = rho * arma::eye(p, p) + lambda2 * V;
  c2 = arma::solve(V, rho * X);
}


void CMM2_update(arma::mat& M, const arma::mat& UX, const arma::sp_mat& U, const arma::mat& UWU, const double& rho, const double& lambda2)
{
  // Input
  // M : matrix of dimensions c times p which we use to minimize the loss
  // X : matrix of dimensions p times p which is the "anchor point" in the minimization. In our case this X = cold - u5 / rho
  // UWU : matrix of dimensions p times p which contains the weights used to reflect importance of clustering i and j
  // rho : scalar, parameter ADMM
  // lambda2 : scalar, regularization parameter clustering

  // Function : Computes the update for c2

  // Output
  // void, c2 is passed by reference and adjusted inside the function

  // Preliminaries
  int c = M.n_rows;
  arma::mat V(c, c, arma::fill::zeros);

  // Fill the matrix V
  for (int i = 0; i < c; i++) {
    for (int j = 0; j < i; j++) {
      if (UWU(i, j) > 0) {
        double temp = UWU(i, j) / std::max(arma::norm(M.row(i) - M.row(j)), 1e-6);

        V(i, i) += temp;
        V(j, j) += temp;
        V(i, j) -= temp;
        V(j, i) -= temp;
      }
    }
  }

  // Compute the update as c2 = (rho * I_pxp + lambda2 * V)^-1 * rho * X
  V = rho * U.t() * U + lambda2 * V;
  M = arma::solve(V, rho * UX);
}


arma::mat CMM1(const arma::mat& cold, const arma::mat& u5, const arma::mat& W, const double& rho, const double& lambda2)
{
  // Input
  // X : matrix of dimensions p times p which is the "anchor point" in the minimization. In our case this X = cold - u5 / rho
  // W : matrix of dimensions p times p which contains the weights used to reflect importance of clustering i and j
  // rho : scalar, parameter ADMM
  // lambda2 : scalar, regularization parameter clustering

  // Function : Computes the update for c2

  // Output
  // c2 : matrix of dimensions p times p which we use to minimize the loss

  // Preliminaries
  //int p = cold.n_rows;
  int t = 0;
  arma::mat X = cold - 1 / rho * u5;
  arma::mat c2(X);

  // Initialize loss values
  double loss1 = cluster_loss(c2, X, W, rho, lambda2);
  double loss0 = 2 * loss1;

  // Set some constants
  const int max_iter = 500;
  const double eps_conv = 1e-7;

  // While the relative decrease in the loss function is above some value and the maximum number of iterations is not reached, update c2
  while (fabs((loss0 - loss1) / loss1) > eps_conv && t < max_iter) {
    CMM1_update(c2, X, W, rho, lambda2);

    loss0 = loss1;
    loss1 = cluster_loss(c2, X, W, rho, lambda2);

    t++;
  }

  return c2;
}


arma::mat CMM2(const arma::mat& cold, const arma::mat& u5, const arma::mat& W, const double& rho, const double& lambda2, const double& eps_fusions, const arma::mat& warm_start)
{
  // Input
  // X : matrix of dimensions p times p which is the "anchor point" in the minimization. In our case this X = cold - u5 / rho
  // W : matrix of dimensions p times p which contains the weights used to reflect importance of clustering i and j
  // rho : scalar, parameter ADMM
  // lambda2 : scalar, regularization parameter clustering

  // Function : Computes the update for c2

  // Output
  // c2 : matrix of dimensions p times p which we use to minimize the loss

  // Preliminaries
  arma::mat X = cold - 1 / rho * u5;
  arma::mat M(warm_start);
  arma::mat UX(X);
  arma::mat UWU(W);
  arma::sp_mat U; U.eye(M.n_rows, M.n_rows);
  int t = 0;
  int n_clusters_old = M.n_rows;
  int n_clusters_new = M.n_rows;
  bool clusters_reduced = false;

  // Set some constants
  const int max_iter = 500;
  const int n_pre_updates = 10;
  const double eps_conv = 1e-7;

  // While the relative decrease in the loss function is above some value and the maximum number of iterations is not reached, update c2
  if (lambda2 > 0) {
    for (int i = 0; i < n_pre_updates; i++) {
      CMM2_update(M, UX, U, UWU, rho, lambda2);
      t++;
    }

    // Initialize loss values
    double loss1 = cluster_loss_fusions(M, X, U, UWU, rho, lambda2);
    double loss0 = 2 * loss1;

    while (fabs((loss0 - loss1) / loss1) > eps_conv && t < max_iter) {
      CMM2_update(M, UX, U, UWU, rho, lambda2);

      find_fusions(M, UWU, U, eps_fusions);
      n_clusters_new = M.n_rows;

      if (n_clusters_new < n_clusters_old) {
        UWU = U.t() * W * U;
        clusters_reduced = true;
      }

      while (n_clusters_new < n_clusters_old) {
        find_fusions(M, UWU, U, eps_fusions);
        n_clusters_old = n_clusters_new;
        n_clusters_new = M.n_rows;

        if (n_clusters_new < n_clusters_old) {
          UWU = U.t() * W * U;
        }
      }

      if (clusters_reduced) {
        UX = U.t() * X;
        clusters_reduced = false;

        loss1 = cluster_loss_fusions(M, X, U, UWU, rho, lambda2);
        loss0 = 2 * loss1;
      } else {
        loss0 = loss1;
        loss1 = cluster_loss_fusions(M, X, U, UWU, rho, lambda2);
      }

      t++;
    }
  }

  arma::mat c2 = U * M;
  return c2;
}






