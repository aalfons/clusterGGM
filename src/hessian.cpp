#include <RcppArmadillo.h>
#include "hessian.h"
#include "utils.h"
#include "variables.h"


double dd_lasso_penalty(double x, double eps)
{
    if (x >= -eps && x <= eps) {
        return 1.0 / eps;
    }

    return 0.0;
}


arma::mat
hessian(const Variables& vars, const arma::mat& RStar0_inv,
        const arma::mat& S, const arma::sp_mat& W_cpath,
        const arma::mat& W_lasso, double lambda_cpath,
        double lambda_lasso, double eps_lasso, int k)
{
    /* Compute the Hessian for cluster k
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
     * Hessian
     */

    // Create references to the variables in the struct
    const arma::mat &R = vars.m_R;
    const arma::vec &A = vars.m_A;
    const arma::ivec &p = vars.m_p;
    const arma::vec &D = vars.m_D;

    // Number of clusters
    int n_clusters = R.n_cols;

    // Initialize result
    arma::mat result(n_clusters + 1, n_clusters + 1);

    // Get r_k to make computations easier
    arma::vec r_k = drop_variable(arma::vec(R.col(k)), k);

    // Store the result of (R^*0)^-1 r_k, as it is used many times
    arma::vec Vr_k = RStar0_inv * r_k;

    // Compute h
    double h = A(k) + (p(k) - 1) * R(k, k) - p(k) * arma::dot(r_k, Vr_k);
    double h2 = square(h);

    // LOG DET PART
    // d/d(a_kk)^2
    double temp_logdet_0 = (p(k) - 1) / square(A(k) - R(k, k));

    result(0, 0) = 1.0 / h2 + temp_logdet_0;

    // d/d(a_kk)d(r_kk)
    result(0, 1 + k) = (p(k) - 1) / h2 - temp_logdet_0;
    result(1 + k, 0) = result(0, 1 + k);

    // d/d(a_kk)d(r_k)
    arma::vec temp_logdet_1 = -2.0 / h2 * p(k) * Vr_k;

    for (int i = 0; i < n_clusters; i++) {
        if (i == k) continue;
        result(0, 1 + i) = temp_logdet_1(i - (i > k));
        result(1 + i, 0) = temp_logdet_1(i - (i > k));
    }

    // d/d(r_kk)^2
    result(k + 1, k + 1) = (p(k) - 1) * (p(k) - 1) / h2 + temp_logdet_0;

    // d/d(r_kk)d(r_k)
    temp_logdet_1 *= (p(k) - 1);

    for (int i = 0; i < n_clusters; i++) {
        if (i == k) continue;
        result(k + 1, 1 + i) = temp_logdet_1(i - (i > k));
        result(1 + i, k + 1) = temp_logdet_1(i - (i > k));
    }

    // d/d(r_k)^2
    double scale_inv = 2.0 * p(k) / h;
    double scale_outer = 4.0 * p(k) * p(k) / h2;

    for (int i = 0; i < n_clusters; i++) {
        if (i == k) continue;

        int i_reduced = i - (i > k);
        double outer_i = scale_outer * Vr_k(i_reduced);

        for (int j = 0; j < k; j++) {
            result(j + 1, 1 + i) =
                scale_inv * RStar0_inv(j, i_reduced) + outer_i * Vr_k(j);
        }

        for (int j = k + 1; j < n_clusters; j++) {
            result(j + 1, 1 + i) =
                scale_inv * RStar0_inv(j - 1, i_reduced) + outer_i * Vr_k(j - 1);
        }
    }

    // r_kk does not occur in the loss function yet if p_k is 1. So, set its
    // second derivative to 1 to keep the Hessian PD.
    if (p(k) == 1) result(k + 1, k + 1) = 1.0;

    // CLUSTERPATH PART
    // Skip if lambda is not positive
    if (lambda_cpath > 0) {
        // Initialize result
        arma::mat H_cpath(n_clusters + 1, n_clusters + 1, arma::fill::zeros);

        // Cluster sizes as a vec, for elementwise use below
        arma::vec p_vec = arma::conv_to<arma::vec>::from(p);

        // Special case for the kth column
        arma::uword e = W_cpath.col_ptrs[k];

        for (auto W_it = W_cpath.begin_col(k); W_it != W_cpath.end_col(k); ++W_it, ++e) {
            // Index
            int l = W_it.row();

            // Compute the inverse of the distance between k and l
            double inv_norm_kl1 = 1.0 / std::max(D(e), 1e-12);
            double inv_norm_kl2 = inv_norm_kl1 * inv_norm_kl1;
            double inv_norm_kl3 = inv_norm_kl2 * inv_norm_kl1;

            // d/d(a_kk)^2
            double temp_cpath_0 = inv_norm_kl1;
            temp_cpath_0 -= square(A(k) - A(l)) * inv_norm_kl3;
            temp_cpath_0 *= (*W_it);
            H_cpath(0, 0) += temp_cpath_0;

            // d/d(a_kk)d(r_kk)
            temp_cpath_0 = -(p(k) - 1) * (*W_it) * inv_norm_kl3;
            temp_cpath_0 *= (A(k) - A(l)) * (R(k, k) - R(k, l));
            H_cpath(k + 1, 0) += temp_cpath_0;
            H_cpath(0, k + 1) += temp_cpath_0;

            // d/d(r_kk)^2
            temp_cpath_0 = -(p(k) - 1) * square(R(k, k) - R(l, k)) * inv_norm_kl3;
            temp_cpath_0 += inv_norm_kl1;
            temp_cpath_0 *= (*W_it) * (p(k) - 1);
            H_cpath(k + 1, k + 1) += temp_cpath_0;

            // Part of d/d(a_kk)d(r_k)
            double temp_cpath_1 = (p(k) - 1) * (R(l, k) - R(k, k));
            temp_cpath_1 += (p(l) - 1) * (R(k, l) - R(l, l));
            temp_cpath_0 = -(*W_it) * inv_norm_kl3 * (A(k) - A(l));
            temp_cpath_0 *= temp_cpath_1;
            H_cpath(0, l + 1) += temp_cpath_0;
            H_cpath(l + 1, 0) += temp_cpath_0;

            // Part of d/d(r_kk)d(r_k)
            temp_cpath_0 = temp_cpath_1 * (R(k, l) - R(k, k)) * inv_norm_kl2;
            temp_cpath_0 = (1 - temp_cpath_0)  * inv_norm_kl1;
            temp_cpath_0 *= -(*W_it) * (p(k) - 1);
            H_cpath(k + 1, l + 1) += temp_cpath_0;
            H_cpath(l + 1, k + 1) += temp_cpath_0;

            // First part of d/d(r_km)^2
            temp_cpath_0 = square(temp_cpath_1) * inv_norm_kl2;
            temp_cpath_0 = (*W_it) * (p(k) + p(l) - 2.0 - temp_cpath_0);
            temp_cpath_0 *= inv_norm_kl1;
            H_cpath(l + 1, l + 1) += temp_cpath_0;

            // Remaining terms range over m (and, for one block, m_p) not equal
            // to k or l. For a fixed pair (k, l), every one of those terms is
            // proportional to u(m) := p(m) * (R(k, m) - R(l, m)), and the m/m_p
            // cross terms are the off-diagonal part of the outer product
            // u * u^T. By zeroing elements k and l the exlusion is handled by
            // u.
            arma::vec u = p_vec % (R.col(k) - R.col(l));
            u(k) = 0.0;
            u(l) = 0.0;

            double scale1 = (*W_it) * inv_norm_kl1;
            double scale3 = (*W_it) * inv_norm_kl3;

            // d/d(r_km)^2 (both parts) and the m/m_p cross terms
            for (int m = 0; m < n_clusters; m++) {
                if (m == k || m == l) continue;

                H_cpath(m + 1, m + 1) += scale1 * p_vec(m);
                H_cpath.col(m + 1).subvec(1, n_clusters) -= (scale3 * u(m)) * u;
            }

            // Remaining parts of d/d(a_kk)d(r_k), d/d(r_kk)d(r_k), and the
            // symmetric l-vs-m entries
            double coef_a = -scale3 * (A(k) - A(l));
            double coef_r = -scale3 * (p(k) - 1) * (R(k, k) - R(k, l));
            double coef_l = -scale3 * temp_cpath_1;

            H_cpath.col(0).subvec(1, n_clusters) += coef_a * u;
            H_cpath.row(0).subvec(1, n_clusters) += coef_a * u.t();

            H_cpath.col(k + 1).subvec(1, n_clusters) += coef_r * u;
            H_cpath.row(k + 1).subvec(1, n_clusters) += coef_r * u.t();

            H_cpath.col(l + 1).subvec(1, n_clusters) += coef_l * u;
            H_cpath.row(l + 1).subvec(1, n_clusters) += coef_l * u.t();
        }

        for (int m = 0; m < (int) W_cpath.n_cols; m++) {
            if (m == k) continue;

            arma::uword e_m = W_cpath.col_ptrs[m];

            for (auto W_it = W_cpath.begin_col(m); W_it != W_cpath.end_col(m); ++W_it, ++e_m) {
                // Index
                int l = W_it.row();

                if (l == k || m == k) continue;

                double inv_norm_ml1 = 1.0 / std::max(D(e_m), 1e-12);
                double inv_norm_ml2 = inv_norm_ml1 * inv_norm_ml1;

                // d/d(r_mm)d(r_mm)
                double temp_cpath_0 = p(k) * square(R(m, k) - R(l, k));
                temp_cpath_0 = 1.0 - temp_cpath_0 * inv_norm_ml2;
                temp_cpath_0 *= p(k) * (*W_it) * inv_norm_ml1;
                H_cpath(m + 1, m + 1) += temp_cpath_0;

                // Hijack this loop for some other computations
                temp_cpath_0 = p(k) * square(R(m, k) - R(l, k));
                temp_cpath_0 = 1.0 - temp_cpath_0 * inv_norm_ml2;
                temp_cpath_0 *= p(k) * (*W_it) * inv_norm_ml1;
                H_cpath(l + 1, m + 1) -= temp_cpath_0;
            }
        }

        // Add Hessian
        result += lambda_cpath * H_cpath;
    }

    // LASSO PART
    // Skip if lambda is not positive
    if (lambda_lasso > 0) {
        // Hessian for the value on the diagonal
        if (p(k) > 1) {
            // Hessian of the penalty function
            double dd_lasso = dd_lasso_penalty(R(k, k), eps_lasso);

            // Add Hessian
            result(1 + k, 1 + k) += lambda_lasso * W_lasso(k, k) * dd_lasso;
        }

        for (int i = 0; i < n_clusters; i++) {
            if (i == k) continue;

            // Hessian of the penalty function for off-diagonal elements
            double dd_lasso = dd_lasso_penalty(R(i, k), eps_lasso);

            // Add Hessian
            result(1 + i, 1 + i) += 2.0 * lambda_lasso * W_lasso(i, k) * dd_lasso;
        }
    }

    return result;
}
