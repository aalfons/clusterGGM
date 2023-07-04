#include <RcppEigen.h>
#include "norms.h"


inline double square(double x)
{
    return x * x;
}


double squaredNormTheta(const Eigen::MatrixXd& Theta, int i, int j)
{
    // Number of rows/columns of theta
    int K = Theta.cols();

    // Initialize result
    double result = square(Theta(i, i) - Theta(j, j));

    // Take the difference between
    for (int k = 0; k < K; k++) {
        if (k == i || k == j) {
            continue;
        }

        result += square(Theta(k, i) - Theta(k, j));
    }

    return result;
}


// [[Rcpp::export]]
double normTheta(const Eigen::MatrixXd& Theta, int i, int j)
{
    return std::sqrt(squaredNormTheta(Theta, i, j));
}


// [[Rcpp::export]]
double normRA(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
              const Eigen::VectorXi& p, int i, int j)
{
    // Number of rows/cols of R
    int K = R.rows();

    // Initialize result
    double result = square(A(i) - A(j));

    for (int k = 0; k < K; k++) {
        if (k == i || k == j) {
            continue;
        }

        result += p[k] * square(R(k, i) - R(k, j));
    }

    result += (p(i) - 1) * square(R(i, i) - R(j, i));
    result += (p(j) - 1) * square(R(j, j) - R(j, i));

    return std::sqrt(result);
}
