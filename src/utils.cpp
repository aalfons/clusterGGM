#include <RcppEigen.h>


void printVector(const Eigen::VectorXd& vec)
{
    Rcpp::Rcout << '[';

    for (int i = 0; i < vec.size() - 1; i++) {
        Rcpp::Rcout << vec(i) << ' ';
    }

    Rcpp::Rcout << vec(vec.size() - 1) << "]\n";
}


void printVector(const Eigen::VectorXi& vec)
{
    Rcpp::Rcout << '[';

    for (int i = 0; i < vec.size() - 1; i++) {
        Rcpp::Rcout << vec(i) << ' ';
    }

    Rcpp::Rcout << vec(vec.size() - 1) << "]\n";
}


void printMatrix(const Eigen::MatrixXd& mat)
{
    Rcpp::Rcout << '[';

    for (int i = 0; i < mat.rows(); i++) {
        if (i != 0) Rcpp::Rcout << ' ';

        for (int j = 0; j < mat.cols() - 1; j++) {
            if (mat(i, j) >= 0) Rcpp::Rcout << ' ';

            Rcpp::Rcout << mat(i, j) << ' ';
        }

        if (i == mat.rows() - 1) Rcpp::Rcout << mat(i, mat.cols() - 1) << "]\n";
        else Rcpp::Rcout << mat(i, mat.cols() - 1) << '\n';
    }
}


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

    printVector(Au);
    printMatrix(result);

    return result;
}
