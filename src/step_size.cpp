#include <RcppEigen.h>
#include <cmath>
#include "utils.h"
#include "loss.h"
#include "step_size.h"


// [[Rcpp::export]]
Eigen::VectorXd maxStepSize(const Eigen::MatrixXd& R,
                            const Eigen::VectorXd& A,
                            const Eigen::VectorXi& p,
                            const Eigen::MatrixXd& R_star_0_inv,
                            const Eigen::VectorXd& g, int k)
{
    // Number of clusters
    int n_clusters = R.cols();

    // Get R[k, -k]
    Eigen::VectorXd r_k = R.row(k);
    r_k = dropVariable(r_k, k);

    // Get parts of the gradient
    double g_a_kk = g(0);
    double g_r_kk = g(1 + k);
    Eigen::VectorXd g_r_k = g.tail(n_clusters);
    g_r_k = dropVariable(g_r_k, k);

    // Compute constants
    Eigen::VectorXd temp0 = r_k.transpose() * R_star_0_inv;
    double c = A(k) + (p(k) - 1) * R(k, k) - p(k) * temp0.dot(r_k);
    double b = -g_a_kk - (p(k) - 1) * g_r_kk + 2 * p(k) * temp0.dot(g_r_k);
    double a = -p(k) * g_r_k.dot(R_star_0_inv * g_r_k);

    // Compute bounds
    double temp1 = std::sqrt(b * b - 4 * a * c);
    double x0 = (-b + temp1) / (2 * a);
    double x1 = (-b - temp1) / (2 * a);

    // Store bounds
    Eigen::VectorXd result(2);
    result(0) = std::min(x0, x1);
    result(1) = std::max(x0, x1);

    // Second part of the log determinant: log(A[k] - R[k, k])
    if (g_a_kk - g_r_kk > 0) {
        result(1) = std::min(result(1), (A(k) - R(k, k)) / (g_a_kk - g_r_kk));
    } else if (g_a_kk - g_r_kk < 0) {
        result(0) = std::max(result(0), (A(k) - R(k, k)) / (g_a_kk - g_r_kk));
    }

    // Add a buffer to compensate for numerical inaccuracies
    result(0) += 1e-12;
    result(1) -= 1e-12;

    return result;
}


// [[Rcpp::export]]
double gssStepSize(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
                   const Eigen::VectorXi& p, const Eigen::VectorXi& u,
                   const Eigen::MatrixXd& R_star_0_inv,
                   const Eigen::MatrixXd& S, const Eigen::MatrixXd& UWU,
                   const Eigen::VectorXd& g, double lambda_cpath, int k,
                   double a, double b, double tol)
{
    // Constants related to the golden ratio
    double invphi = (std::sqrt(5) - 1) / 2;      // 1 / phi
    double invphi2 = (3 - std::sqrt(5)) / 2;     // 1 / phi^2

    // Interval size
    double h = b - a;

    // Required steps to achieve tolerance
    int n_steps = std::ceil(std::log(tol / h) / std::log(invphi));

    // Midpoints
    double c = a + invphi2 * h;
    double d = a + invphi * h;

    // Compute loss for step size c
    auto [R_update, A_update] = updateRA(R, A, -c * g, k);
    double yc = lossRAk(R_update, A_update, p, u, R_star_0_inv, S, UWU, lambda_cpath, k);

    // Compute loss for step size d
    updateRAInplace(R_update, A_update, -(d - c) * g, k);
    double yd = lossRAk(R_update, A_update, p, u, R_star_0_inv, S, UWU, lambda_cpath, k);

    // Reset R_update and A_update
    updateRAInplace(R_update, A_update, d * g, k);

    for (int i = 0; i < n_steps; i++) {
        if (yc < yd) {
            b = d;
            d = c;
            yd = yc;
            h = invphi * h;
            c = a + invphi2 * h;

            // Compute new loss value
            updateRAInplace(R_update, A_update, -c * g, k);
            yc = lossRAk(R_update, A_update, p, u, R_star_0_inv, S, UWU, lambda_cpath, k);
            updateRAInplace(R_update, A_update, c * g, k);
        } else {
            a = c;
            c = d;
            yc = yd;
            h = invphi * h;
            d = a + invphi * h;

            // Compute new loss value
            updateRAInplace(R_update, A_update, -d * g, k);
            yd = lossRAk(R_update, A_update, p, u, R_star_0_inv, S, UWU, lambda_cpath, k);
            updateRAInplace(R_update, A_update, d * g, k);
        }
    }

    // Return step size
    if (yc < yd) {
        return (a + d) / 2;
    }

    return (c + b) / 2;
}
