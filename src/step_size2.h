#include <RcppEigen.h>
#include "partial_loss_constants.h"
#include "variables.h"


Eigen::VectorXd maxStepSize2(const Variables& vars,
                             const Eigen::MatrixXd& Rstar0_inv,
                             const Eigen::VectorXd& d, int k);

double stepSizeSelection(const Variables& vars,
                         const PartialLossConstants& consts,
                         const Eigen::MatrixXd& Rstar0_inv,
                         const Eigen::MatrixXd& S,
                         const Eigen::SparseMatrix<double>& W,
                         const Eigen::VectorXd& ddir, double lambda, int k,
                         double lo, double hi, double tol);
