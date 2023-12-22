#include <RcppEigen.h>
#include "norms.h"



Eigen::MatrixXd weights_Theta(const Eigen::MatrixXd& Theta, double phi)
{
    // Number of cols/rows
    int n = Theta.cols();

    // Initialize result
    Eigen::MatrixXd result(n, n);

    // Mean squared norm
    double msn = 0;

    // Fill result
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            if (i == j) {
                result(i, j) = 0;
                continue;
            }

            // Compute squared norm Theta
            double snt = squared_norm_Theta(Theta, i, j);

            // Fill in weight matrix
            result(i, j) = snt;
            result(j, i) = snt;

            // Add to sum of squared norms
            msn += snt;
        }
    }

    // Mean squared norm
    msn /= double(n * n - n) / 2.0;

    // Scale distances
    result /= msn;

    // Apply Gaussian weights formula
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (i == j) continue;

            result(i, j) = std::exp(-phi * result(i, j));
        }
    }

    return result;
}


// [[Rcpp::export(.scaled_squared_norms)]]
Eigen::MatrixXd scaled_squared_norms(const Eigen::MatrixXd& Theta)
{
    // Number of cols/rows
    int n = Theta.cols();

    // Initialize result
    Eigen::MatrixXd result(n, n);

    // Mean squared norm
    double msn = 0;

    // Fill result
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            if (i == j) {
                result(i, j) = 0;
                continue;
            }

            // Compute squared norm Theta
            double snt = squared_norm_Theta(Theta, i, j);

            // Fill in matrix
            result(i, j) = snt;
            result(j, i) = snt;

            // Add to sum of squared norms
            msn += snt;
        }
    }

    // Mean squared norm
    msn /= double(n * n - n) / 2.0;

    // Scale squared distances
    if (msn > 0) result /= msn;

    return result;
}


// [[Rcpp::export(.squared_norms)]]
Eigen::MatrixXd squared_norms(const Eigen::MatrixXd& Theta)
{
    // Number of cols/rows
    int n = Theta.cols();

    // Initialize result
    Eigen::MatrixXd result(n, n);

    // Fill result
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            if (i == j) {
                result(i, j) = 0;
                continue;
            }

            // Compute squared norm Theta
            double snt = squared_norm_Theta(Theta, i, j);

            // Fill in matrix
            result(i, j) = snt;
            result(j, i) = snt;
        }
    }

    return result;
}
