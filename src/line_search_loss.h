#ifndef LINESEARCHLOSS_H
#define LINESEARCHLOSS_H

#include <RcppArmadillo.h>
#include "loss.h"
#include "partial_loss_constants.h"
#include "utils.h"
#include "variables.h"


struct LineSearchLoss {
    /* The partial loss for cluster k as a function of the step size taken
     * along a fixed descent direction.
     *
     * Moving along the direction changes only row/column k of R and element k
     * of A, and it changes them linearly in the step size. Every term of the
     * partial loss is therefore a polynomial in the step size or the square
     * root of one, so the whole function is described by a fixed set of
     * coefficients. Those are computed once, after which each candidate step
     * size costs one pass over the clusterpath edges and the lasso weights.
     */

    // Determinant part: log(m_det0 + s * m_det1 + s^2 * m_det2) plus
    // m_pk_minus_1 * log(m_diff0 + s * m_diff1)
    double m_det0, m_det1, m_det2;
    double m_diff0, m_diff1;
    double m_pk_minus_1;

    // Covariance part: m_cov0 + s * m_cov1
    double m_cov0, m_cov1;

    // Clusterpath part: sum over the edges of
    // m_cpath_w % sqrt(m_cpath_a + s * m_cpath_b + s^2 * m_cpath_c)
    arma::vec m_cpath_w, m_cpath_a, m_cpath_b, m_cpath_c;

    // Lasso part: sum over the clusters of
    // m_lasso_w % lasso_penalty(m_lasso_r + s * m_lasso_d)
    arma::vec m_lasso_w, m_lasso_r, m_lasso_d;

    double m_lambda_cpath, m_lambda_lasso, m_eps_lasso;

    LineSearchLoss(const Variables& vars, const PartialLossConstants& consts,
                   const arma::mat& Rstar0_inv,
                   const arma::sp_mat& W_cpath,
                   const arma::mat& W_lasso, const arma::vec& ddir,
                   double lambda_cpath, double lambda_lasso, double eps_lasso,
                   int k)
    {
        /* Inputs:
         * vars: struct containing the optimization variables
         * consts: struct containing the optimization constants
         * Rstar0_inv: inverse of R* excluding row/column k
         * W_cpath: sparse weight matrix
         * W_lasso: lasso weight matrix
         * ddir: descent direction
         * lambda_cpath: clusterpath regularization parameter
         * lambda_lasso: lasso regularization parameter
         * eps_lasso: width of the quadratic approximation of the lasso penalty
         * k: cluster of interest
         */

        // Create references to the variables in the structs
        const arma::mat &R = vars.m_R;
        const arma::vec &A = vars.m_A;
        const arma::ivec &p = vars.m_p;
        const arma::vec &E = consts.m_E;

        // Number of clusters
        int n_clusters = R.n_cols;

        m_lambda_cpath = lambda_cpath;
        m_lambda_lasso = lambda_lasso;
        m_eps_lasso = eps_lasso;
        m_pk_minus_1 = p(k) - 1;

        // Change in A[k] and in column k of R per unit of step size
        double d_akk = ddir(0);
        arma::vec d_r = ddir.tail(n_clusters);
        double d_rkk = d_r(k);

        // Current value and direction of R[-k, k]
        arma::vec r_k = R.col(k);
        drop_variable_inplace(r_k, k);
        arma::vec v_k = d_r;
        drop_variable_inplace(v_k, k);

        // Coefficients of the quadratic form r_k' Rstar0_inv r_k. Computing
        // the two products here is what removes a matrix-vector product from
        // every evaluation
        arma::vec M_r = Rstar0_inv * r_k;
        arma::vec M_v = Rstar0_inv * v_k;
        double q0 = arma::dot(r_k, M_r);
        double q1 = arma::dot(r_k, M_v) + arma::dot(v_k, M_r);
        double q2 = arma::dot(v_k, M_v);

        // Determinant part
        m_det0 = A(k) + m_pk_minus_1 * R(k, k) - p(k) * q0;
        m_det1 = d_akk + m_pk_minus_1 * d_rkk - p(k) * q1;
        m_det2 = -p(k) * q2;

        m_diff0 = A(k) - R(k, k);
        m_diff1 = d_akk - d_rkk;

        // Covariance part
        m_cov0 = 2 * arma::dot(r_k, consts.m_uSU) + consts.m_uSu * R(k, k)
                 + m_diff0 * consts.m_pTraceS;
        m_cov1 = 2 * arma::dot(v_k, consts.m_uSU) + consts.m_uSu * d_rkk
                 + m_diff1 * consts.m_pTraceS;

        // Clusterpath part, skip if lambda is not positive
        if (lambda_cpath > 0) {
            // One edge per pair of clusters with a nonzero weight
            int n_edges = 0;
            for (int j = 0; j < (int) W_cpath.n_cols; j++) {
                for (auto W_it = W_cpath.begin_col(j);
                     W_it != W_cpath.end_col(j); ++W_it) {
                    if ((int) W_it.row() > j) n_edges++;
                }
            }

            m_cpath_w.set_size(n_edges);
            m_cpath_a.set_size(n_edges);
            m_cpath_b.set_size(n_edges);
            m_cpath_c.set_size(n_edges);

            int e = 0;

            for (int j = 0; j < (int) W_cpath.n_cols; j++) {
                arma::uword nz = W_cpath.col_ptrs[j];

                for (auto W_it = W_cpath.begin_col(j);
                     W_it != W_cpath.end_col(j); ++W_it, ++nz) {
                    // Index
                    int i = W_it.row();

                    // Skip loop for half of the computations
                    if (i <= j) continue;

                    m_cpath_w(e) = *W_it;

                    if (i == k || j == k) {
                        // The squared distance between cluster k and the other
                        // cluster is a weighted sum of squares of linear
                        // functions of the step size
                        int l = (i == k) ? j : i;

                        double a = 0, b = 0, c = 0;

                        auto add_square = [&](double weight, double base,
                                              double slope) {
                            a += weight * base * base;
                            b += 2.0 * weight * base * slope;
                            c += weight * slope * slope;
                        };

                        add_square(1.0, A(k) - A(l), d_akk);

                        for (int m = 0; m < n_clusters; m++) {
                            if (m == k || m == l) continue;

                            add_square(p(m), R(m, k) - R(m, l), d_r(m));
                        }

                        add_square(p(k) - 1, R(k, k) - R(l, k), d_rkk - d_r(l));
                        add_square(p(l) - 1, R(l, l) - R(l, k), -d_r(l));

                        m_cpath_a(e) = a;
                        m_cpath_b(e) = b;
                        m_cpath_c(e) = c;
                    } else {
                        // Only the difference between R(i, k) and R(j, k)
                        // depends on the step size, the rest of the squared
                        // distance is held in E
                        double base = R(i, k) - R(j, k);
                        double slope = d_r(i) - d_r(j);

                        m_cpath_a(e) = E(nz) + p(k) * base * base;
                        m_cpath_b(e) = 2.0 * p(k) * base * slope;
                        m_cpath_c(e) = p(k) * slope * slope;
                    }

                    e++;
                }
            }
        }

        // Lasso part, skip if lambda is not positive
        if (lambda_lasso > 0) {
            m_lasso_w.set_size(n_clusters);

            for (int i = 0; i < n_clusters; i++) {
                // The diagonal element is counted once, off-diagonal elements
                // are counted for both triangles
                m_lasso_w(i) = (i == k) ? W_lasso(k, k) : 2.0 * W_lasso(i, k);
            }

            m_lasso_r = R.col(k);
            m_lasso_d = d_r;
        }
    }

    double value(double s) const
    {
        /* Compute the partial loss for a step size
         *
         * Inputs:
         * s: step size
         *
         * Output:
         * The partial loss
         */

        // Determinant part
        double loss_det = std::log(m_det0 + s * (m_det1 + s * m_det2));
        loss_det += m_pk_minus_1 * std::log(m_diff0 + s * m_diff1);

        // Covariance part
        double loss_cov = m_cov0 + s * m_cov1;

        // Clusterpath part
        double loss_cpath = 0;

        for (arma::uword e = 0; e < m_cpath_w.n_elem; e++) {
            // The squared distance is nonnegative in exact arithmetic, the
            // maximum guards against a rounding error turning it negative
            double d2 = m_cpath_a(e) + s * (m_cpath_b(e) + s * m_cpath_c(e));

            loss_cpath += m_cpath_w(e) * std::sqrt(std::max(d2, 0.0));
        }

        // Lasso part
        double loss_lasso = 0;

        for (arma::uword i = 0; i < m_lasso_w.n_elem; i++) {
            loss_lasso += m_lasso_w(i) *
                lasso_penalty(m_lasso_r(i) + s * m_lasso_d(i), m_eps_lasso);
        }

        return -loss_det + loss_cov + m_lambda_cpath * loss_cpath
               + m_lambda_lasso * loss_lasso;
    }
};

#endif // LINESEARCHLOSS_H
