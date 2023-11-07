#ifndef NORMS_H
#define NORMS_H

#include <RcppEigen.h>


double normRA2(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
               const Eigen::VectorXi& p, int i, int j);

#endif // NORMS_H
