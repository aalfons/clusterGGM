#include <RcppEigen.h>
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
                         const Eigen::MatrixXd& Rstar0_inv,
                         const Eigen::MatrixXd& S,
                         const Eigen::SparseMatrix<double>& W,
                         const Eigen::VectorXd& d, double lambda, int k,
                         double lo, double hi, double tol)
{
    // Create references to the variables in the struct
    const Eigen::MatrixXd &R = vars.m_R;
    const Eigen::MatrixXd &A = vars.m_A;
    const Eigen::VectorXi &p = vars.m_p;

    // Check on the inputs
    if (hi <= lo) {
        return 0.0;
    }

    // Copy the distance matrix, it serves as a starting point for computing
    // distances after modifying one row/column of R
    Eigen::SparseMatrix<double> E(vars.m_D);

    for (int j = 0; j < E.outerSize(); j++) {
        // Iterator
        Eigen::SparseMatrix<double>::InnerIterator E_it(E, j);

        for (; W_it; ++W_it) {
            // In this loop, compute D^2 - p(k) * (R(i, k) - R(j, k)) for all
            // i and j not equal to k. This allows all distances between i and j
            // (again not equal to k) to be calculated much faster, as only 5
            // additional flops are required to compute the new distance
            // instead of O(n_clusters) flops
            if (i == k || j == k) continue;

            // Index
            i = E_it.row();

            // Part that has to be subtracted
            double sub = p(k) * square2(R(i, k) - R(j, k));
            E_it.valueRef() = square2(E_it.value()) - sub;
        }
    }

    // Constants related to the golden ratio
    double invphi1 = (std::sqrt(5) - 1) / 2;      // 1 / phi
    double invphi2 = (3 - std::sqrt(5)) / 2;      // 1 / phi^2

    // Initialize a and b
    double a = lo;
    double b = hi;

    // Interval size
    double h = b - a;

    // Required steps to achieve tolerance
    int n_steps = std::ceil(std::log(tol / h) / std::log(invphi));

    // Midpoints c and d
    double c = a + invphi2 * h;
    double d = a + invphi * h;

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

/*
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

    // Compute loss for step size 0
    double y0 = lossRAk(R, A, p, u, R_star_0_inv, S, UWU, lambda_cpath, k);

    // Candidate step size
    double s = 0.0;
    if (yc < yd) {
        s = (a + d) / 2.0;
    } else {
        s = (c + b) / 2.0;
    }

    // Compute new loss value
    updateRAInplace(R_update, A_update, -s * g, k);
    double ys = lossRAk(R_update, A_update, p, u, R_star_0_inv, S, UWU, lambda_cpath, k);

    // If candidate step size s is not at least better than step size of 0,
    // return 0, else return s
    if (y0 <= ys) return 0.0;*/
    return 0.0;
}
