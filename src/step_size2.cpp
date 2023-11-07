#include <RcppEigen.h>
#include "loss2.h"
#include "partial_loss_constants.h"
#include "step_size2.h"
#include "utils2.h"
#include "variables.h"


Eigen::VectorXd maxStepSize2(const Variables& vars,
                             const Eigen::MatrixXd& Rstar0_inv,
                             const Eigen::VectorXd& d, int k)
{
    /* Compute the interval for the step size that keeps the result positive
     * definite.
     *
     * Computations are done using the negative descent direction (-d) due to
     * previous versions of this code using the gradient.
     *
     * Inputs:
     * vars: struct containing the optimization variables
     * Rstar0_inv: inverse of R* excluding row/column k
     * d: descent direction
     * k: cluster of interest
     *
     * Output:
     * Vector with the minimum and maximum step sizes
     */

    // Create references to the variables in the struct
    const Eigen::MatrixXd &R = vars.m_R;
    const Eigen::MatrixXd &A = vars.m_A;
    const Eigen::VectorXi &p = vars.m_p;

    // Number of clusters
    int n_clusters = R.cols();

    // Vector that holds result
    Eigen::VectorXd result(2);

    // Get parts of the descent direction
    double d_a_kk = -d(0);
    double d_r_kk = -d(1 + k);

    if (n_clusters > 1) {
        // Get R[k, -k] and its descent direction
        Eigen::VectorXd r_k = R.row(k);
        dropVariableInplace2(r_k, k);
        Eigen::VectorXd d_r_k = -d.tail(n_clusters);
        dropVariableInplace2(d_r_k, k);

        // Compute constants
        Eigen::VectorXd temp0 = r_k.transpose() * Rstar0_inv;
        double c = A(k) + (p(k) - 1) * R(k, k) - p(k) * temp0.dot(r_k);
        double b = -d_a_kk - (p(k) - 1) * d_r_kk + 2 * p(k) * temp0.dot(d_r_k);
        double a = -p(k) * d_r_k.dot(Rstar0_inv * d_r_k);

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
        double b = d_a_kk + (p(k) - 1) * d_r_kk;

        if (b > 0) {
            result(1) = std::min(result(1), a / b);
        } else if (b < 0) {
            result(0) = std::max(result(0), a / b);
        }
    }

    // Second part of the log determinant: log(A[k] - R[k, k])
    if (d_a_kk - d_r_kk > 0) {
        result(1) = std::min(result(1), (A(k) - R(k, k)) / (d_a_kk - d_r_kk));
    } else if (d_a_kk - d_r_kk < 0) {
        result(0) = std::max(result(0), (A(k) - R(k, k)) / (d_a_kk - d_r_kk));
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


double stepSizeSelection(const Variables& vars,
                         const PartialLossConstants& consts,
                         const Eigen::MatrixXd& Rstar0_inv,
                         const Eigen::MatrixXd& S,
                         const Eigen::SparseMatrix<double>& W,
                         const Eigen::VectorXd& ddir, double lambda, int k,
                         double lo, double hi, double tol)
{
    /* Damped Newton
    # f: Objective function
    # gradient_f: Gradient of the objective function
    # x: Current point in the optimization space
    # direction: Search direction
    # alpha: A parameter in the range (0, 1) for controlling the step size
    # beta: A parameter in the range (0, 1) for controlling the reduction factor
    # max_iterations: Maximum number of iterations allowed

         t = 1.0  # Initial step size
         iterations = 0

         while iterations < max_iterations:
    # Evaluate the objective function at the new point
         new_x = x + t * direction
         fx = f(new_x)

    # Calculate the expected reduction (Armijo condition)
         expected_reduction = alpha * t * gradient_f(x).dot(direction)

         if fx <= f(x) + expected_reduction:
    # Sufficient reduction achieved, accept the step
         return t
         else:
    # Reduce the step size and continue searching
         t = beta * t
         iterations += 1

    # If we reach here, the search did not converge in max_iterations
         return t
     */
    // Check on the inputs
    if (hi <= lo) {
        return 0.0;
    }

    // Create references to the variables in the struct
    const Eigen::MatrixXd &R = vars.m_R;
    const Eigen::MatrixXd &A = vars.m_A;
    const Eigen::VectorXi &p = vars.m_p;

    // Number of clusters
    int n_clusters = R.cols();

    // Constants related to the golden ratio
    double invphi1 = (std::sqrt(5) - 1) / 2;      // 1 / phi
    double invphi2 = (3 - std::sqrt(5)) / 2;      // 1 / phi^2

    // Initialize a and b
    double a = lo;
    double b = hi;

    // Interval size
    double h = b - a;

    // Required steps to achieve tolerance
    int n_steps = std::ceil(std::log(tol / h) / std::log(invphi1));

    // Midpoints c and d
    double c = a + invphi2 * h;
    double d = a + invphi1 * h;

    // Compute loss for step size c
    auto [R_update, A_update] = updateRA2(R, A, c * ddir, k);
    double yc = lossPartial(
        vars, consts, R_update, A_update, Rstar0_inv, S, W, lambda, k
    );

    // Compute loss for step size d
    updateRAInplace2(R_update, A_update, (d - c) * ddir, k);
    double yd = lossPartial(
        vars, consts, R_update, A_update, Rstar0_inv, S, W, lambda, k
    );

    // Reset R_update and A_update
    updateRAInplace2(R_update, A_update, -d * ddir, k);

    for (int i = 0; i < n_steps; i++) {
        if (yc < yd) {
            b = d;
            d = c;
            yd = yc;
            h = invphi1 * h;
            c = a + invphi2 * h;

            // Compute new loss value
            updateRAInplace2(R_update, A_update, c * ddir, k);
            yc = lossPartial(
                vars, consts, R_update, A_update, Rstar0_inv, S, W, lambda, k
            );
            updateRAInplace2(R_update, A_update, -c * ddir, k);
        } else {
            a = c;
            c = d;
            yc = yd;
            h = invphi1 * h;
            d = a + invphi1 * h;

            // Compute new loss value
            updateRAInplace2(R_update, A_update, d * ddir, k);
            yd = lossPartial(
                vars, consts, R_update, A_update, Rstar0_inv, S, W, lambda, k
            );
            updateRAInplace2(R_update, A_update, -d * ddir, k);
        }
    }

    // Compute loss for step size 0
    double y0 = lossPartial(
        vars, consts, R_update, A_update, Rstar0_inv, S, W, lambda, k
    );

    // Candidate step size
    double s = 0.0;
    if (yc < yd) {
        s = (a + d) / 2.0;
    } else {
        s = (c + b) / 2.0;
    }

    // Compute new loss value
    updateRAInplace2(R_update, A_update, s * ddir, k);
    double ys = lossPartial(
        vars, consts, R_update, A_update, Rstar0_inv, S, W, lambda, k
    );

    // If candidate step size s is not at least better than step size of 0,
    // return 0, else return s
    if (y0 <= ys) return 0.0;
    return s;
}
