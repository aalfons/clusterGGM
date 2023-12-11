#include <RcppEigen.h>
#include "variables.h"


Eigen::VectorXd
gradient(const Variables& vars, const Eigen::MatrixXd& Rstar0_inv,
         const Eigen::MatrixXd& S, const Eigen::SparseMatrix<double>& W,
         double lambda, int k);
