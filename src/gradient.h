#include <RcppArmadillo.h>
#include "variables.h"


arma::vec
gradient(const Variables& vars, const arma::mat& Rstar0_inv,
         const arma::mat& S, const arma::sp_mat& W_cpath,
         const arma::mat& W_lasso, double lambda_cpath,
         double lambda_lasso, double eps_lasso, int k);
