#include <RcppEigen.h>
#include "norms.h"
#include "utils.h"
#include "gradient.h"


// [[Rcpp::export]]
Eigen::VectorXd gradient(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
                         const Eigen::VectorXi& p, const Eigen::VectorXi& u,
                         const Eigen::MatrixXd& R_star_0_inv,
                         const Eigen::MatrixXd& S, const Eigen::MatrixXd& UWU,
                         double lambda_cpath, int k, int fuse_candidate)
{
    // Number of rows/columns of R
    int K = R.cols();

    // Number of rows/columns of S
    int P = S.cols();

    // Initialize result;
    Eigen::VectorXd result(K + 1);

    // Log determinant part
    Eigen::VectorXd r_k = dropVariable(R.col(k), k);

    // Some temporary variables
    Eigen::VectorXd temp_log_0 = R_star_0_inv * r_k;
    double temp_log_1 = A(k) + (p(k) - 1) * R(k, k);
    temp_log_1 -= p(k) * r_k.dot(temp_log_0);

    // Gradient for A[k]
    result(0) = -1.0 / temp_log_1 - (p(k) - 1) / (A(k) - R(k, k));

    // Gradient for R[k, k]
    result(1 + k) = -(p(k) - 1) / temp_log_1 + (p(k) - 1) / (A(k) - R(k, k));

    // Temporary vector to store gradient of R[k, -k] in
    Eigen::VectorXd grad_r_k = 2 * p(k) / temp_log_1 * temp_log_0;

    // Fill in the gradient for R[k, -k]
    for (int i = 0; i < K; i++) {
        if (i == k) continue;

        result(1 + i) = grad_r_k(i - (i > k));
    }

    // Covariance part
    // Gradient for A[k]
    double covGradientAk = partialTrace(S, u, k);
    result(0) += covGradientAk;

    // Gradient for R[k, k]
    result(1 + k) += sumSelectedElements(S, u, p, k) - covGradientAk;

    // Gradient for R[k, -k]
    grad_r_k = 2 * sumMultipleSelectedElements(S, u, p, k);

    // Fill in the gradient for R[k, -k]
    for (int i = 0; i < K; i++) {
        if (i == k) continue;

        result(1 + i) += grad_r_k(i - (i > k));
    }

    // Return if lambda is not positive
    if (lambda_cpath <= 0) {
        return result;
    }

    // Clusterpath part
    double grad_a_kk = 0;
    double grad_r_kk = 0;
    grad_r_k = Eigen::VectorXd::Zero(K);

    for (int l = 0; l < K; l++) {
        if (l == k || l == fuse_candidate) continue;

        double inv_norm_kl = 1.0 / std::max(normRA(R, A, p, k, l), 1e-12);

        grad_a_kk += (A(k) - A(l)) * UWU(k, l) * inv_norm_kl;
        grad_r_kk += (R(k, k) - R(k, l)) * (p(k) - 1) * UWU(k, l) * inv_norm_kl;

        for (int m = 0; m < K; m++) {
            if (m == l) continue;

            grad_r_k(m) += UWU(k, l) * inv_norm_kl * (R(k, m) - R(m, l)) * p(m);
        }
    }

    for (int m = 0; m < K; m++) {
        if (m == k) continue;

        for (int l = 0; l < K; l++) {
            if (l == m || l == k) continue;

            double inv_norm_ml = 1.0 / std::max(normRA(R, A, p, m, l), 1e-12);

            grad_r_k(m) += UWU(m, l) * inv_norm_ml * (R(k, m) - R(k, l)) * p(k);
        }

        if (m != fuse_candidate && m != k) {
            double inv_norm_km = 1.0 / std::max(normRA(R, A, p, k, m), 1e-12);

            double temp = (p(m) - 1) * (R(k, m) - R(m, m));
            temp += (p(k) - 1) * (R(k, m) - R(k, k));
            grad_r_k(m) += temp * inv_norm_km * UWU(k, m);
        }
    }

    // Add to the gradient again
    result(0) += lambda_cpath * grad_a_kk;
    result(1 + k) += lambda_cpath * grad_r_kk;
    for (int i = 0; i < K; i++) {
        if (i == k) continue;
        result(1 + i) += lambda_cpath * grad_r_k(i);
    }

    return result;
}
