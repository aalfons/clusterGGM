#ifndef NORMS_H
#define NORMS_H

#include <RcppEigen.h>

double squared_norm_Theta(const Eigen::MatrixXd& Theta, int i, int j);

double norm_RA(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
               const Eigen::VectorXi& p, int i, int j);

#endif // NORMS_H
