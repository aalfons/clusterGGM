#include <RcppEigen.h>
#include "norms.h"


// [[Rcpp::export]]
Eigen::MatrixXd weightsTheta(const Eigen::MatrixXd& Theta, double phi)
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

            result(i, j) = std::exp(-phi * squaredNormTheta(Theta, i, j));
            result(j, i) = result(i, j);
        }
    }

    return result;
}
