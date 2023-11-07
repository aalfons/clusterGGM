#include <RcppEigen.h>
#include "norms2.h"
#include "utils2.h"


double normRA2(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
               const Eigen::VectorXi& p, int i, int j)
{
    // Number of rows/cols of R
    int K = R.rows();

    // Initialize result
    double result = square2(A(i) - A(j));

    for (int k = 0; k < K; k++) {
        if (k == i || k == j) {
            continue;
        }

        result += p[k] * square2(R(k, i) - R(k, j));
    }

    result += (p(i) - 1) * square2(R(i, i) - R(j, i));
    result += (p(j) - 1) * square2(R(j, j) - R(j, i));

    return std::sqrt(result);
}
