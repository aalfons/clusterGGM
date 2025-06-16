#include <RcppEigen.h>
#include "partial_loss_constants.h"
#include "variables.h"


Eigen::VectorXd max_step_size(const Variables& vars,
                              const Eigen::MatrixXd& Rstar0_inv,
                              const Eigen::VectorXd& d, int k);

double step_size_gss(const Variables& vars, const PartialLossConstants& consts,
                     const Eigen::MatrixXd& Rstar0_inv,
                     const Eigen::MatrixXd& S,
                     const Eigen::SparseMatrix<double>& W_cpath,
                     const Eigen::MatrixXd& W_lasso,
                     const Eigen::VectorXd& ddir, double lambda_cpath,
                     double lambda_lasso, double eps_lasso, int k, double lo,
                     double hi, double tol);
