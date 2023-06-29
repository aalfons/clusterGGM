#include <RcppEigen.h>


// [[Rcpp::export]]
Eigen::MatrixXd updateInverse(const Eigen::MatrixXd& inverse, const Eigen::VectorXd& vec, int i)
{
    // Sherman-Morrison with A = inverse, u = e_i, v = vec
    Eigen::VectorXd Au = inverse.col(i);
    Eigen::VectorXd vA = vec.transpose() * inverse;
    Eigen::MatrixXd N = Au * vA.transpose();
    double D = 1.0 / (1.0 + vA(i));

    Eigen::MatrixXd result = inverse - D * N;

    // Sherman-Morrison with A = result, u = vec, v = e_i
    Au = result * vec;
    vA = result.row(i);
    N = Au * vA.transpose();
    D = 1.0 / (1.0 + Au(i));

    result -= D * N;

    return result;
}
