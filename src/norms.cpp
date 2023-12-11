#include <RcppEigen.h>
#include "norms.h"
#include "utils.h"


double squared_norm_Theta(const Eigen::MatrixXd& Theta, int i, int j)
{
    // Number of rows/columns of theta
    int K = Theta.cols();

    // Initialize result
    double result = square(Theta(i, i) - Theta(j, j));

    // Fill result
    for (int k = 0; k < K; k++) {
        if (k == i || k == j) {
            continue;
        }
        result += square(Theta(k, i) - Theta(k, j));
    }
    return result;
}


double norm_RA(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
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
