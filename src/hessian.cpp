#include <RcppEigen.h>
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


Eigen::MatrixXd
hessian(const Variables& vars, const Eigen::MatrixXd& RStar0_inv,
        const Eigen::MatrixXd& S, const Eigen::SparseMatrix<double>& W_cpath,
        const Eigen::MatrixXd& W_lasso, double lambda_cpath,
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
    const Eigen::MatrixXd &R = vars.m_R;
    const Eigen::MatrixXd &A = vars.m_A;
    const Eigen::VectorXi &p = vars.m_p;
    const Eigen::SparseMatrix<double> &D = vars.m_D;

    // Number of clusters
    int n_clusters = R.cols();

    // Initialize result
    Eigen::MatrixXd result(n_clusters + 1, n_clusters + 1);

    // Get r_k to make computations easier
    Eigen::VectorXd r_k = drop_variable(R.col(k), k);

    // Store the result of (R^*0)^-1 r_k, as it is used many times
    Eigen::VectorXd Vr_k = RStar0_inv * r_k;

    // Compute h
    double h = A(k) + (p(k) - 1) * R(k, k) - p(k) * r_k.dot(Vr_k);
    double h2 = square(h);

    // LOG DET PART
    // d/d(a_kk)^2
    double temp_logdet_0 = (p(k) - 1) / square(A(k) - R(k, k));

    result(0, 0) = 1.0 / h2 + temp_logdet_0;

    // d/d(a_kk)d(r_kk)
    result(0, 1 + k) = (p(k) - 1) / h2 - temp_logdet_0;
    result(1 + k, 0) = result(0, 1 + k);

    // d/d(a_kk)d(r_k)
    Eigen::VectorXd temp_logdet_1 = -2.0 / h2 * p(k) * Vr_k;

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
    Eigen::MatrixXd temp_logdet_2 = 2.0 * p(k) * RStar0_inv;
    temp_logdet_2 += + 4.0 * p(k) * p(k) / h * Vr_k * Vr_k.transpose();
    temp_logdet_2 /= h;

    for (int i = 0; i < n_clusters; i++) {
        if (i == k) continue;
        for (int j = 0; j < n_clusters; j++) {
            if (j == k) continue;
            result(j + 1, 1 + i) = temp_logdet_2(j - (j > k), i - (i > k));
        }
    }

    // Catch the issue of row/column k completely filled with zeros due to the
    // cluster size of cluster k being 1, causing r_kk not to occur in the loss
    // function yet
    if (p(k) == 1) result(k + 1, k + 1) = 1.0;

    // CLUSTERPATH PART
    // Skip if lambda is not positive
    if (lambda_cpath > 0) {
        // Initialize result
        Eigen::MatrixXd H_cpath =
            Eigen::MatrixXd::Zero(n_clusters + 1, n_clusters + 1);

        // Special case for the kth column
        Eigen::SparseMatrix<double>::InnerIterator D_it(D, k);
        Eigen::SparseMatrix<double>::InnerIterator W_it(W_cpath, k);

        for (; W_it; ++W_it) {
            // Index
            int l = W_it.row();

            // Compute the inverse of the distance between k and l
            double inv_norm_kl1 = 1.0 / std::max(D_it.value(), 1e-12);
            double inv_norm_kl2 = inv_norm_kl1 * inv_norm_kl1;
            double inv_norm_kl3 = inv_norm_kl2 * inv_norm_kl1;

            // d/d(a_kk)^2
            double temp_cpath_0 = inv_norm_kl1;
            temp_cpath_0 -= square(A(k) - A(l)) * inv_norm_kl3;
            temp_cpath_0 *= W_it.value();
            H_cpath(0, 0) += temp_cpath_0;

            // d/d(a_kk)d(r_kk)
            temp_cpath_0 = -(p(k) - 1) * W_it.value() * inv_norm_kl3;
            temp_cpath_0 *= (A(k) - A(l)) * (R(k, k) - R(k, l));
            H_cpath(k + 1, 0) += temp_cpath_0;
            H_cpath(0, k + 1) += temp_cpath_0;

            // d/d(r_kk)^2
            temp_cpath_0 = -(p(k) - 1) * square(R(k, k) - R(l, k)) * inv_norm_kl3;
            temp_cpath_0 += inv_norm_kl1;
            temp_cpath_0 *= W_it.value() * (p(k) - 1);
            H_cpath(k + 1, k + 1) += temp_cpath_0;

            // Part of d/d(a_kk)d(r_k)
            double temp_cpath_1 = (p(k) - 1) * (R(l, k) - R(k, k));
            temp_cpath_1 += (p(l) - 1) * (R(k, l) - R(l, l));
            temp_cpath_0 = -W_it.value() * inv_norm_kl3 * (A(k) - A(l));
            temp_cpath_0 *= temp_cpath_1;
            H_cpath(0, l + 1) += temp_cpath_0;
            H_cpath(l + 1, 0) += temp_cpath_0;

            // Part of d/d(r_kk)d(r_k)
            temp_cpath_0 = temp_cpath_1 * (R(k, l) - R(k, k)) * inv_norm_kl2;
            temp_cpath_0 = (1 - temp_cpath_0)  * inv_norm_kl1;
            temp_cpath_0 *= -W_it.value() * (p(k) - 1);
            H_cpath(k + 1, l + 1) += temp_cpath_0;
            H_cpath(l + 1, k + 1) += temp_cpath_0;

            // First part of d/d(r_km)^2
            temp_cpath_0 = square(temp_cpath_1) * inv_norm_kl2;
            temp_cpath_0 = W_it.value() * (p(k) + p(l) - 2.0 - temp_cpath_0);
            temp_cpath_0 *= inv_norm_kl1;
            H_cpath(l + 1, l + 1) += temp_cpath_0;

            // Loop over m, as many elements require the distance between k and l
            for (int m = 0; m < n_clusters; m++) {
                if (m == k || m == l) continue;

                // Remaining part of d/d(a_kk)d(r_k)
                temp_cpath_0 = -W_it.value() * inv_norm_kl3 * p(m) * (A(k) - A(l));
                temp_cpath_0 *= (R(k, m) - R(l, m));
                H_cpath(0, m + 1) += temp_cpath_0;
                H_cpath(m + 1, 0) += temp_cpath_0;

                // Remaining part of d/d(r_kk)d(r_k)
                temp_cpath_0 = -W_it.value() * inv_norm_kl3 * p(m) * (p(k) - 1);
                temp_cpath_0 *= ((R(k, m) - R(m, l)) * (R(k, k) - R(k, l)));
                H_cpath(k + 1, m + 1) += temp_cpath_0;
                H_cpath(m + 1, k + 1) += temp_cpath_0;

                // Second part of d/d(r_km)^2
                temp_cpath_0 = -p(m) * square(R(k, m) - R(l, m)) * inv_norm_kl2;
                temp_cpath_0 += 1.0;
                temp_cpath_0 *= p(m) * W_it.value() * inv_norm_kl1;
                H_cpath(m + 1, m + 1) += temp_cpath_0;

                // Hijack this loop for some other calculations as well
                temp_cpath_0 = (p(k) - 1) * (R(l, k) - R(k, k));
                temp_cpath_0 += (p(l) - 1) * (R(k, l) - R(l, l));
                temp_cpath_0 *= W_it.value() * p(m) * (R(k, m) - R(l, m));
                temp_cpath_0 *= inv_norm_kl3;
                H_cpath(l + 1, m + 1) -= temp_cpath_0;

                // Hijack this loop for some other calculations as well
                temp_cpath_0 = (p(k) - 1) * (R(l, k) - R(k, k));
                temp_cpath_0 += (p(l) - 1) * (R(k, l) - R(l, l));
                temp_cpath_0 *= W_it.value() * p(m) * (R(k, m) - R(l, m));
                temp_cpath_0 *= inv_norm_kl3;
                H_cpath(m + 1, l + 1) -= temp_cpath_0;

                // Further hijacking
                for (int m_p = 0; m_p < n_clusters; m_p++) {
                    if (m_p == k || m_p == m || m_p == l) continue;

                    temp_cpath_0 = p(m) * p(m_p) * (R(k, m) - R(l, m));
                    temp_cpath_0 *= R(k, m_p) - R(l, m_p);
                    temp_cpath_0 *= W_it.value() * inv_norm_kl3;
                    H_cpath(m + 1, m_p + 1) -= temp_cpath_0;
                }
            }

            // Continue iterator for D
            ++D_it;
        }

        for (int m = 0; m < W_cpath.outerSize(); m++) {
            if (m == k) continue;

            // Iterators
            Eigen::SparseMatrix<double>::InnerIterator D_it(D, m);
            Eigen::SparseMatrix<double>::InnerIterator W_it(W_cpath, m);

            for (; W_it; ++W_it) {
                // Index
                int l = W_it.row();

                if (l == k || m == k) {
                    // Continue iterator for D
                    ++D_it;
                    continue;
                }

                double inv_norm_ml1 = 1.0 / std::max(D_it.value(), 1e-12);
                double inv_norm_ml2 = inv_norm_ml1 * inv_norm_ml1;

                // d/d(r_mm)d(r_mm)
                double temp_cpath_0 = p(k) * square(R(m, k) - R(l, k));
                temp_cpath_0 = 1.0 - temp_cpath_0 * inv_norm_ml2;
                temp_cpath_0 *= p(k) * W_it.value() * inv_norm_ml1;
                H_cpath(m + 1, m + 1) += temp_cpath_0;

                // Hijack this loop for some other computations
                temp_cpath_0 = p(k) * square(R(m, k) - R(l, k));
                temp_cpath_0 = 1.0 - temp_cpath_0 * inv_norm_ml2;
                temp_cpath_0 *= p(k) * W_it.value() * inv_norm_ml1;
                H_cpath(l + 1, m + 1) -= temp_cpath_0;

                // Continue iterator for D
                ++D_it;
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
