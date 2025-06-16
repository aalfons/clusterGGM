#include <RcppEigen.h>
#include "variables.h"


Eigen::VectorXd
gradient(const Variables& vars, const Eigen::MatrixXd& Rstar0_inv,
         const Eigen::MatrixXd& S, const Eigen::SparseMatrix<double>& W_cpath,
         const Eigen::MatrixXd& W_lasso, double lambda_cpath,
         double lambda_lasso, double eps_lasso, int k);
