#ifndef UTILS_H
#define UTILS_H

#include <RcppEigen.h>


double square2(double x);

Eigen::SparseMatrix<double>
convertToSparse(const Eigen::MatrixXd& W_keys, const Eigen::VectorXd& W_values,
                int n_variables);

#endif // UTILS_H
