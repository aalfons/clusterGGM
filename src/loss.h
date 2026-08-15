#ifndef LOSS_H
#define LOSS_H

#include <RcppArmadillo.h>
#include "variables.h"


double lasso_penalty(double x, double eps);

double loss_complete(const Variables& vars, const arma::mat& S,
                     const arma::sp_mat& W_cpath,
                     const arma::mat& W_lasso, double lambda_cpath,
                     double lambda_lasso, double lasso_eps);

#endif // LOSS_H
