#include <RcppEigen.h>
#include "variables.h"


Eigen::MatrixXd
hessian2(const Variables& vars, const Eigen::MatrixXd& RStar0_inv,
         const Eigen::MatrixXd& S, const Eigen::SparseMatrix<double>& W,
         double lambda, int k);
