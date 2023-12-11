#ifndef LOSS_H
#define LOSS_H

#include <RcppEigen.h>
#include "partial_loss_constants.h"
#include "variables.h"


double loss_complete(const Variables& vars, const Eigen::MatrixXd& S,
                     const Eigen::SparseMatrix<double>& W, double lambda_cpath);

double loss_partial(const Variables vars, const PartialLossConstants& consts,
                    const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
                    const Eigen::MatrixXd& Rstar0_inv, const Eigen::MatrixXd& S,
                    const Eigen::SparseMatrix<double>& W, double lambda, int k);

#endif // LOSS_H

