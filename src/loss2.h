#include <RcppEigen.h>
#include "variables.h"


double lossComplete(const Variables& vars, const Eigen::VectorXi& p,
                    const Eigen::VectorXi& u, const Eigen::MatrixXd& S,
                    const Eigen::SparseMatrix<double>& W, double lambda_cpath);
