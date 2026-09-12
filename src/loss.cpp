#include <RcppArmadillo.h>
#include "utils.h"
#include "variables.h"


double lasso_penalty(double x, double eps)
{
    if (x >= -eps && x <= eps) {
        return x * x / (2.0 * eps) + eps / 2.0;
    }

    return std::fabs(x);
}


double loss_complete(const Variables& vars, const arma::mat& S,
                     const arma::sp_mat& W_cpath,
                     const arma::mat& W_lasso, double lambda_cpath,
                     double lambda_lasso, double lasso_eps)
{
    /* Compute the value of the entire loss function, including all variables
     *
     * Inputs:
     * vars: struct containing the optimization variables
     * S: sample covariance matrix
     * W_cpath: sparse weight matrix
     * lambda_cpath: regularization parameter
     *
     * Output:
     * The loss
     */

    // Create references to the variables in the struct
    const arma::mat &R = vars.m_R;
    const arma::vec &A = vars.m_A;
    const arma::ivec &p = vars.m_p;
    const arma::ivec &u = vars.m_u;
    const arma::vec &D = vars.m_D;

    // Number of clusters
    int n_clusters = R.n_cols;

    // Compute log determinant
    arma::mat Rstar = vars.m_Rstar;
    for (int i = 0; i < (int) R.n_cols; i++) {
        Rstar.row(i) *= std::sqrt((double) p(i));
        Rstar.col(i) *= std::sqrt((double) p(i));
    }
    double loss_det = std::log(arma::det(Rstar));

    for (int i = 0; i < n_clusters; i++) {
        loss_det += (p(i) - 1) * std::log(A(i) - R(i, i));
    }

    // Covariance part of the loss: tr(S * Theta), where Theta is R and A
    // expanded to the full n_variables x n_variables scale via u
    double loss_cov = arma::accu(S % compute_Theta(R, A, u));

    // Clusterpath part
    double loss_cpath = 0;

    // Skip if lambda is not positive
    if (lambda_cpath > 0) {
        for (int i = 0; i < (int) W_cpath.n_cols; i++) {
            arma::uword e = W_cpath.col_ptrs[i];

            for (auto W_it = W_cpath.begin_col(i); W_it != W_cpath.end_col(i); ++W_it, ++e) {
                if (i > (int) W_it.row()) {
                    loss_cpath += (*W_it) * D(e);
                }
            }
        }
    }

    // Lasso part
    double loss_lasso = 0;

    // Skip if lambda is not positive
    if (lambda_lasso > 0) {
        for (int j = 0; j < (int) W_lasso.n_cols; j++) {
            // Off-diagonal elements
            for (int i = 0; i < j; i++) {
                loss_lasso += 2.0 * W_lasso(i, j) * lasso_penalty(R(i, j), lasso_eps);
            }

            // Diagonal elements
            loss_lasso += W_lasso(j, j) * lasso_penalty(R(j, j), lasso_eps);
        }
    }


    return -loss_det + loss_cov + lambda_cpath * loss_cpath + lambda_lasso * loss_lasso;
}
