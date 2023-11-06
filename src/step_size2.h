#include <RcppEigen.h>
#include "variables.h"


Eigen::VectorXd maxStepSize2(const Variables& vars,
                             const Eigen::MatrixXd& Rstar0_inv,
                             const Eigen::VectorXd& d, int k);
