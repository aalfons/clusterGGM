#include <RcppArmadillo.h>
#include "norms.h"



// [[Rcpp::export(.scaled_squared_norms)]]
arma::mat scaled_squared_norms(const arma::mat& Theta)
{
    // Number of cols/rows
    int n = Theta.n_cols;

    // Initialize result
    arma::mat result(n, n);

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
arma::mat squared_norms(const arma::mat& Theta)
{
    // Number of cols/rows
    int n = Theta.n_cols;

    // Initialize result
    arma::mat result(n, n);

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
