#include <RcppEigen.h>
#include <cmath>
#include "utils.h"
#include "loss.h"
#include "step_size.h"
#include "clock.h"


// [[Rcpp::export]]
Eigen::VectorXd maxStepSize(const Eigen::MatrixXd& R,
                            const Eigen::VectorXd& A,
                            const Eigen::VectorXi& p,
                            const Eigen::MatrixXd& R_star_0_inv,
                            const Eigen::VectorXd& g, int k)
{
    // Number of clusters
    int n_clusters = R.cols();

    // Vector that holds result
    Eigen::VectorXd result(2);

    // Get parts of the gradient
    double g_a_kk = g(0);
    double g_r_kk = g(1 + k);

    if (n_clusters > 1) {
        // Get R[k, -k] and its gradient
        Eigen::VectorXd r_k = R.row(k);
        r_k = dropVariable(r_k, k);
        Eigen::VectorXd g_r_k = g.tail(n_clusters);
        g_r_k = dropVariable(g_r_k, k);

        // Compute constants
        Eigen::VectorXd temp0 = r_k.transpose() * R_star_0_inv;
        double c = A(k) + (p(k) - 1) * R(k, k) - p(k) * temp0.dot(r_k);
        double b = -g_a_kk - (p(k) - 1) * g_r_kk + 2 * p(k) * temp0.dot(g_r_k);
        double a = -p(k) * g_r_k.dot(R_star_0_inv * g_r_k);

        // Compute bounds
        double temp1 = std::sqrt(std::max(b * b - 4 * a * c, 0.0));
        double x0 = (-b + temp1) / std::min(2 * a, -1e-12);
        double x1 = (-b - temp1) / std::min(2 * a, -1e-12);

        // Store bounds
        result(0) = std::min(x0, x1);
        result(1) = std::max(x0, x1);
    } else {
        result(0) = -10.0;
        result(1) = 10.0;

        double a = A(k) + (p(k) - 1) * R(k, k);
        double b = g_a_kk + (p(k) - 1) * g_r_kk;

        if (b > 0) {
            result(1) = std::min(result(1), a / b);
        } else if (b < 0) {
            result(0) = std::max(result(0), a / b);
        }
    }

    // Second part of the log determinant: log(A[k] - R[k, k])
    if (g_a_kk - g_r_kk > 0) {
        result(1) = std::min(result(1), (A(k) - R(k, k)) / (g_a_kk - g_r_kk));
    } else if (g_a_kk - g_r_kk < 0) {
        result(0) = std::max(result(0), (A(k) - R(k, k)) / (g_a_kk - g_r_kk));
    }

    // Add a buffer to compensate for numerical inaccuracies
    result(0) += 1e-12;
    result(1) -= 1e-12;

    // Lastly, check if the upper bound is smaller than zero
    if (result(1) < 0) {
        result(1) = 0;
    }

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
    // Check on the inputs
    if (b <= a) {
        return 0;
    }

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
    CLOCK.tick("cggm - gradientDescent - gssStepSize - lossRAk");
    double yc = lossRAk(R_update, A_update, p, u, R_star_0_inv, S, UWU, lambda_cpath, k);
    CLOCK.tock("cggm - gradientDescent - gssStepSize - lossRAk");

    // Compute loss for step size d
    updateRAInplace(R_update, A_update, -(d - c) * g, k);
    CLOCK.tick("cggm - gradientDescent - gssStepSize - lossRAk");
    double yd = lossRAk(R_update, A_update, p, u, R_star_0_inv, S, UWU, lambda_cpath, k);
    CLOCK.tock("cggm - gradientDescent - gssStepSize - lossRAk");

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
            CLOCK.tick("cggm - gradientDescent - gssStepSize - lossRAk");
            yc = lossRAk(R_update, A_update, p, u, R_star_0_inv, S, UWU, lambda_cpath, k);
            CLOCK.tock("cggm - gradientDescent - gssStepSize - lossRAk");
            updateRAInplace(R_update, A_update, c * g, k);
        } else {
            a = c;
            c = d;
            yc = yd;
            h = invphi * h;
            d = a + invphi * h;

            // Compute new loss value
            updateRAInplace(R_update, A_update, -d * g, k);
            CLOCK.tick("cggm - gradientDescent - gssStepSize - lossRAk");
            yd = lossRAk(R_update, A_update, p, u, R_star_0_inv, S, UWU, lambda_cpath, k);
            CLOCK.tock("cggm - gradientDescent - gssStepSize - lossRAk");
            updateRAInplace(R_update, A_update, d * g, k);
        }
    }

    // Compute loss for step size 0
    CLOCK.tick("cggm - gradientDescent - gssStepSize - lossRAk");
    double y0 = lossRAk(R, A, p, u, R_star_0_inv, S, UWU, lambda_cpath, k);
    CLOCK.tock("cggm - gradientDescent - gssStepSize - lossRAk");

    // Candidate step size
    double s = 0.0;
    if (yc < yd) {
        s = (a + d) / 2.0;
    } else {
        s = (c + b) / 2.0;
    }

    // Compute new loss value
    updateRAInplace(R_update, A_update, -s * g, k);
    CLOCK.tick("cggm - gradientDescent - gssStepSize - lossRAk");
    double ys = lossRAk(R_update, A_update, p, u, R_star_0_inv, S, UWU, lambda_cpath, k);
    CLOCK.tock("cggm - gradientDescent - gssStepSize - lossRAk");

    // If candidate step size s is not at least better than step size of 0,
    // return 0, else return s
    if (y0 <= ys) return 0.0;
    return s;
}
