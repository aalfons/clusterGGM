#include <RcppEigen.h>
#include "variables.h"


Eigen::VectorXd
gradient2(const Variables& vars, const Eigen::MatrixXd& Rstar0_inv,
          const Eigen::MatrixXd& S, const Eigen::SparseMatrix<double>& W,
          double lambda, int k);
