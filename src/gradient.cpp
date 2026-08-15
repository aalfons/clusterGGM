#include <RcppArmadillo.h>
#include "gradient.h"
#include "utils.h"
#include "variables.h"


double d_lasso_penalty(double x, double eps)
{
    if (x >= -eps && x <= eps) {
        return x / eps;
    }

    return (x > 0) - (x < 0);
}



arma::vec
gradient(const Variables& vars, const arma::mat& Rstar0_inv,
         const arma::mat& S, const arma::sp_mat& W_cpath,
         const arma::mat& W_lasso, double lambda_cpath,
         double lambda_lasso, double eps_lasso, int k)
{
    /* Compute the gradient for cluster k
     *
     * Inputs:
     * vars: struct containing the optimization variables
     * Rstar0_inv: inverse of R* excluding row/column k
     * S: sample covariance matrix
     * W_cpath: sparse weight matrix
     * lambda_cpath: regularization parameter
     * k: cluster of interest
     *
     * Output:
     * Gradient
     */

    // Create references to the variables in the struct
    const arma::mat &R = vars.m_R;
    const arma::vec &A = vars.m_A;
    const arma::ivec &p = vars.m_p;
    const arma::ivec &u = vars.m_u;
    const arma::vec &D = vars.m_D;

    // Number of clusters
    int n_clusters = R.n_cols;

    // Initialize result;
    arma::vec result(n_clusters + 1);

    // Log determinant part
    arma::vec r_k = drop_variable(arma::vec(R.col(k)), k);

    // Some temporary variables
    arma::vec temp_log_0 = Rstar0_inv * r_k;
    double temp_log_1 = A(k) + (p(k) - 1) * R(k, k);
    temp_log_1 -= p(k) * arma::dot(r_k, temp_log_0);

    // Gradient for A[k]
    result(0) = -1.0 / temp_log_1 - (p(k) - 1) / (A(k) - R(k, k));

    // Gradient for R[k, k]
    result(1 + k) = -(p(k) - 1) / temp_log_1 + (p(k) - 1) / (A(k) - R(k, k));

    // Temporary vector to store gradient of R[k, -k] in
    arma::vec grad_r_k = 2 * p(k) / temp_log_1 * temp_log_0;

    // Fill in the gradient for R[k, -k]
    for (int i = 0; i < n_clusters; i++) {
        if (i == k) continue;

        result(1 + i) = grad_r_k(i - (i > k));
    }

    // Covariance part
    // Gradient for A[k]
    double cov_grad_a_kk = partial_trace(S, u, k);
    result(0) += cov_grad_a_kk;

    // Gradient for R[k, k]
    result(1 + k) += sum_selected_elements(S, u, p, k) - cov_grad_a_kk;

    // Gradient for R[k, -k]
    grad_r_k = 2 * sum_multiple_selected_elements(S, u, p, k);

    // Fill in the gradient for R[k, -k]
    for (int i = 0; i < n_clusters; i++) {
        if (i == k) continue;

        result(1 + i) += grad_r_k(i - (i > k));
    }

    // Skip if lambda is not positive
    if (lambda_cpath > 0) {
        // Clusterpath part
        double grad_a_kk = 0;
        double grad_r_kk = 0;
        grad_r_k = arma::vec(n_clusters, arma::fill::zeros);

        // Special case for the kth column
        arma::uword e = W_cpath.col_ptrs[k];

        for (auto W_it = W_cpath.begin_col(k); W_it != W_cpath.end_col(k); ++W_it, ++e) {
            // index
            int l = W_it.row();

            // Compute the inverse of the distance between k and l
            double inv_norm_kl = 1.0 / std::max(D(e), 1e-12);

            // Add to gradients
            grad_a_kk += (A(k) - A(l)) * (*W_it) * inv_norm_kl;
            grad_r_kk +=
                (R(k, k) - R(k, l)) * (p(k) - 1) * (*W_it) * inv_norm_kl;

            for (int m = 0; m < n_clusters; m++) {
                if (m == l) continue;

                grad_r_k(m) +=
                    (*W_it) * inv_norm_kl * (R(k, m) - R(m, l)) * p(m);
            }

            double temp = (p(l) - 1) * (R(k, l) - R(l, l));
            temp += (p(k) - 1) * (R(k, l) - R(k, k));
            grad_r_k(l) += temp * inv_norm_kl * (*W_it);
        }


        for (int m = 0; m < (int) W_cpath.n_cols; m++) {
            if (m == k) continue;

            arma::uword e_m = W_cpath.col_ptrs[m];

            for (auto W_it = W_cpath.begin_col(m); W_it != W_cpath.end_col(m); ++W_it, ++e_m) {
                // index
                int l = W_it.row();

                if (l == k) continue;

                // Compute the inverse of the distance between m and l
                double inv_norm_ml = 1.0 / std::max(D(e_m), 1e-12);

                // Add to the gradient
                grad_r_k(m) +=
                    (*W_it) * inv_norm_ml * (R(k, m) - R(k, l)) * p(k);
            }
        }

        // Add to the complete gradient again
        result(0) += lambda_cpath * grad_a_kk;
        result(1 + k) += lambda_cpath * grad_r_kk;
        for (int i = 0; i < n_clusters; i++) {
            if (i == k) continue;
            result(1 + i) += lambda_cpath * grad_r_k(i);
        }
    }

    // Skip if lambda is not positive
    if (lambda_lasso > 0) {
        // Gradient for the value on the diagonal
        if (p(k) > 1) {
            // Gradient of the penalty function
            double d_lasso = d_lasso_penalty(R(k, k), eps_lasso);

            // Add gradient
            result(1 + k) += lambda_lasso * W_lasso(k, k) * d_lasso;
        }

        for (int i = 0; i < n_clusters; i++) {
            if (i == k) continue;

            // Gradient of the penalty function for off-diagonal elements
            double d_lasso = d_lasso_penalty(R(i, k), eps_lasso);

            // Add gradient
            result(1 + i) += 2.0 * lambda_lasso * W_lasso(i, k) * d_lasso;
        }
    }

    return result;
}
