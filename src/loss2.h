#include <RcppEigen.h>
#include "partial_loss_constants.h"
#include "variables.h"


double lossComplete(const Variables& vars, const Eigen::MatrixXd& S,
                    const Eigen::SparseMatrix<double>& W, double lambda_cpath);

double lossPartial(const Variables vars, const PartialLossConstants& consts,
                   const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
                   const Eigen::MatrixXd& Rstar0_inv, const Eigen::MatrixXd& S,
                   const Eigen::SparseMatrix<double>& W, double lambda, int k);
