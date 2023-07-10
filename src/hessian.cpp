#include <RcppEigen.h>
#include "norms.h"
#include "utils.h"
#include "gradient.h"
#include "hessian.h"


// [[Rcpp::export]]
Eigen::MatrixXd hessian(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
                        const Eigen::VectorXi& p, const Eigen::VectorXi& u,
                        const Eigen::MatrixXd& R_star_0_inv,
                        const Eigen::MatrixXd& S, const Eigen::MatrixXd& UWU,
                        double lambda, int k)
{
    // Number of rows/columns of R
    int K = R.cols();

    // Initialize result
    Eigen::MatrixXd result(K + 1, K + 1);

    // Get r_k to make computations easier
    Eigen::VectorXd r_k = dropVariable(R.col(k), k);

    // Store the result of (R^*0)^-1 r_k, as it is used many times
    Eigen::VectorXd Vr_k = R_star_0_inv * r_k;

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

    for (int i = 0; i < K; i++) {
        if (i == k) continue;
        result(0, 1 + i) = temp_logdet_1(i - (i > k));
        result(1 + i, 0) = temp_logdet_1(i - (i > k));
    }

    // d/d(r_kk)^2
    result(k + 1, k + 1) = (p(k) - 1) * (p(k) - 1) / h2 + temp_logdet_0;

    // d/d(r_kk)d(r_k)
    temp_logdet_1 *= (p(k) - 1);

    for (int i = 0; i < K; i++) {
        if (i == k) continue;
        result(k + 1, 1 + i) = temp_logdet_1(i - (i > k));
        result(1 + i, k + 1) = temp_logdet_1(i - (i > k));
    }

    // d/d(r_k)^2
    Eigen::MatrixXd temp_logdet_2 = 2.0 * p(k) * R_star_0_inv;
    temp_logdet_2 += + 4.0 * p(k) * p(k) / h * Vr_k * Vr_k.transpose();
    temp_logdet_2 /= h;

    for (int i = 0; i < K; i++) {
        if (i == k) continue;
        for (int j = 0; j < K; j++) {
            if (j == k) continue;
            result(j + 1, 1 + i) = temp_logdet_2(j - (j > k), i - (i > k));
        }
    }

    // CLUSTERPATH PART
    // Initialize result
    Eigen::MatrixXd H_cpath = Eigen::MatrixXd::Zero(K + 1, K + 1);

    for (int l = 0; l < K; l ++) {
        if (l == k) continue;

        // Distance between k and l
        double d_kl = std::max(normRA(R, A, p, k, l), 1e-12);
        double d_kl2 = square(d_kl);
        double d_kl3 = d_kl2 * d_kl;

        // d/d(a_kk)^2
        H_cpath(0, 0) += UWU(k, l) * (1.0 - square(A(k) - A(l)) / d_kl2) / d_kl;

        // d/d(a_kk)/d(r_kk)
        double temp_cpath_0 = -UWU(k, l) * (R(k, k) - R(k, l)) * (A(k) - A(l));
        temp_cpath_0 *= (p(k) - 1) / d_kl3;
        H_cpath(k + 1, 0) += temp_cpath_0;
        H_cpath(0, k + 1) += temp_cpath_0;

        // d/d(r_kk)^2
        temp_cpath_0 = (1.0 - (p(k) - 1) * square(R(k, k) - R(l, k)) / d_kl2);
        temp_cpath_0 *= (p(k) - 1) * UWU(k, l) / d_kl;
        H_cpath(k + 1, k + 1) += temp_cpath_0;

        for (int m = 0; m < K; m++) {
            if (m == k) continue;

            if (l != m) {
                // d/d(a_kk)d(r_k)
                temp_cpath_0 = -UWU(k, l) * p(m) * (R(k, m) - R(l, m));
                temp_cpath_0 *= (A(k) - A(l)) / d_kl3;
                H_cpath(0, m + 1) += temp_cpath_0;
                H_cpath(m + 1, 0) += temp_cpath_0;

                // d/d(r_kk)/d(r_k)
                temp_cpath_0 = -UWU(k, l) * p(m) * (p(k) - 1);
                temp_cpath_0 *= (R(k, m) - R(m, l)) * (R(k, k) - R(k, l));
                temp_cpath_0 /= d_kl3;
                H_cpath(k + 1, m + 1) += temp_cpath_0;
                H_cpath(m + 1, k + 1) += temp_cpath_0;

                // d/d(r_km)^2
                temp_cpath_0 = 1.0 - p(m) * square(R(k, m) - R(l, m)) / d_kl2;
                temp_cpath_0 *= p(m) * UWU(k, l) / d_kl;
                H_cpath(m + 1, m + 1) += temp_cpath_0;

                double d_ml = std::max(normRA(R, A, p, m, l), 1e-12);
                double d_ml2 = square(d_ml);
                temp_cpath_0 = 1.0 - p(k) * square(R(m , k) - R(l, k)) / d_ml2;
                temp_cpath_0 *= p(k) * UWU(m, l) / d_ml;
                H_cpath(m + 1, m + 1) += temp_cpath_0;
            } else {
                // l = m
                // d/d(a_kk)d(r_k)
                double temp_cpath_1 = (p(k) - 1) * (R(m, k) - R(k, k));
                temp_cpath_1 += (p(m) - 1) * (R(k, m) - R(m, m));
                temp_cpath_0 = -UWU(k, m) * temp_cpath_1 * (A(k) - A(m)) / d_kl3;
                H_cpath(0, m + 1) += temp_cpath_0;
                H_cpath(m + 1, 0) += temp_cpath_0;

                // d/d(r_kk)d(r_k)
                temp_cpath_0 = temp_cpath_1 * (R(k, m) - R(k, k)) / d_kl2;
                temp_cpath_0 = (1 - temp_cpath_0) / d_kl;
                temp_cpath_0 *= -UWU(k, m) * (p(k) - 1);
                H_cpath(k + 1, m + 1) += temp_cpath_0;
                H_cpath(m + 1, k + 1) += temp_cpath_0;

                // d/d(r_km)^2
                temp_cpath_0 = square(temp_cpath_1) / d_kl2;
                temp_cpath_0 = UWU(k, m) * (p(k) + p(m) - 2.0 - temp_cpath_0);
                temp_cpath_0 /= d_kl;
                H_cpath(m + 1, m + 1) += temp_cpath_0;
            }
        }

        for (int m_p = 0; m_p < K; m_p++) {
            if (m_p == k) continue;

            for (int m = 0; m < K; m++) {
                if (l == m && m != m_p) {
                    // Now l = m, so d_km = d_kl
                    temp_cpath_0 = (p(k) - 1) * (R(m, k) - R(k, k));
                    temp_cpath_0 += (p(m) - 1) * (R(k, m) - R(m, m));
                    temp_cpath_0 *= UWU(k, m) * p(m_p) * (R(k, m_p) - R(m, m_p));
                    temp_cpath_0 /= d_kl3;
                    H_cpath(m + 1, m_p + 1) -= temp_cpath_0;

                    double d_mmp = std::max(normRA(R, A, p, m, m_p), 1e-12);
                    double d_mmp2 = square(d_mmp);
                    temp_cpath_0 = 1.0 - p(k) * square(R(m_p, k) - R(m, k)) / d_mmp2;
                    temp_cpath_0 *= p(k) * UWU(m, m_p) / d_mmp;
                    H_cpath(m + 1, m_p + 1) -= temp_cpath_0;
                }

                if (m == m_p || l == m || m == k) {
                    continue;
                }

                if (m_p != l) {
                    temp_cpath_0 = p(m) * p(m_p) * (R(k, m) - R(l, m));
                    temp_cpath_0 *= R(k, m_p) - R(l, m_p);
                    temp_cpath_0 *= UWU(k, l) / d_kl3;
                    H_cpath(m + 1, m_p + 1) -= temp_cpath_0;
                } else {
                    temp_cpath_0 = (p(k) - 1) * (R(m_p, k) - R(k, k));
                    temp_cpath_0 += (p(m_p) - 1) * (R(k, m_p) - R(m_p, m_p));
                    temp_cpath_0 *= UWU(k, m_p) * p(m) * (R(k, m) - R(m_p, m));
                    temp_cpath_0 /= d_kl3;
                    H_cpath(m + 1, m_p + 1) -= temp_cpath_0;
                }
            }
        }
    }

    return result + lambda * H_cpath;
}
