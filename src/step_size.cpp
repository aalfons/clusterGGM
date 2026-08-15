#include <RcppArmadillo.h>
#include "line_search_loss.h"
#include "partial_loss_constants.h"
#include "step_size.h"
#include "utils.h"
#include "variables.h"


arma::vec max_step_size(const Variables& vars,
                        const arma::mat& Rstar0_inv,
                        const arma::vec& d, int k)
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
    const arma::mat &R = vars.m_R;
    const arma::vec &A = vars.m_A;
    const arma::ivec &p = vars.m_p;

    // Number of clusters
    int n_clusters = R.n_cols;

    // Vector that holds result
    arma::vec result(2);

    // Get parts of the descent direction
    double d_a_kk = -d(0);
    double d_r_kk = -d(1 + k);

    if (n_clusters > 1) {
        // Get R[k, -k] and its descent direction
        arma::vec r_k = R.row(k).t();
        drop_variable_inplace(r_k, k);
        arma::vec d_r_k = -d.tail(n_clusters);
        drop_variable_inplace(d_r_k, k);

        // Compute constants
        arma::vec temp0 = (r_k.t() * Rstar0_inv).t();
        double c = A(k) + (p(k) - 1) * R(k, k) - p(k) * arma::dot(temp0, r_k);
        double b = -d_a_kk - (p(k) - 1) * d_r_kk + 2 * p(k) * arma::dot(temp0, d_r_k);
        double a = -p(k) * arma::dot(d_r_k, Rstar0_inv * d_r_k);

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


double step_size_gss(const Variables& vars, const PartialLossConstants& consts,
                     const arma::mat& Rstar0_inv,
                     const arma::sp_mat& W_cpath,
                     const arma::mat& W_lasso,
                     const arma::vec& ddir, double lambda_cpath,
                     double lambda_lasso, double eps_lasso, int k, double lo,
                     double hi, double tol)
{
    /* Perform step size selection based on an interval [lo, hi].
     *
     * Inputs:
     * vars: struct containing the optimization variables
     * consts: struct containing the optimization constants
     * Rstar0_inv: inverse of R* excluding row/column k
     * W_cpath: sparse weight matrix
     * ddir: descent direction
     * lambda_cpath: regularization parameter
     * k: cluster of interest
     * lo: lower bound for the step size
     * hi: upper bound for the step size
     * tol: tolerance between lo and hi for terminating the algorithm
     */
    // Check on the inputs
    if (hi <= lo) {
        return 0.0;
    }

    // Express the partial loss as a function of the step size
    LineSearchLoss loss(
        vars, consts, Rstar0_inv, W_cpath, W_lasso, ddir, lambda_cpath,
        lambda_lasso, eps_lasso, k
    );

    // Compute loss for step size 0
    double y0 = loss.value(0.0);

    // Constants related to the golden ratio
    double invphi1 = (std::sqrt(5) - 1) / 2;      // 1 / phi
    double invphi2 = (3 - std::sqrt(5)) / 2;      // 1 / phi^2

    // Initialize a and b
    double a = lo;
    double b = hi;

    // Interval size
    double h = b - a;

    // Number of steps for absolute reduction of interval size, always do a
    // minimum of two steps
    int n_steps = std::ceil(std::log(tol / h) / std::log(invphi1));
    n_steps = std::max(n_steps, 2);

    // Midpoints c and d
    double c = a + invphi2 * h;
    double d = a + invphi1 * h;

    // Compute loss for step sizes c and d
    double yc = loss.value(c);
    double yd = loss.value(d);

    for (int i = 0; i < n_steps; i++) {
        if (yc < yd) {
            b = d;
            d = c;
            yd = yc;
            h = invphi1 * h;
            c = a + invphi2 * h;

            // Compute new loss value
            yc = loss.value(c);
        } else {
            a = c;
            c = d;
            yc = yd;
            h = invphi1 * h;
            d = a + invphi1 * h;

            // Compute new loss value
            yd = loss.value(d);
        }
    }

    // Candidate step size
    double s = 0.0;
    if (yc < yd) {
        s = (a + d) / 2.0;
    } else {
        s = (c + b) / 2.0;
    }

    // Compute new loss value
    double ys = loss.value(s);

    // If candidate step size s is not at least better than step size of 0,
    // return 0, else return s
    if (y0 <= ys) return 0.0;
    return s;
}
