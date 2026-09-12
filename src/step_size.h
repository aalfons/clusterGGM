#include <RcppArmadillo.h>
#include "partial_loss_constants.h"
#include "variables.h"


arma::vec max_step_size(const Variables& vars,
                        const arma::mat& Rstar0_inv,
                        const arma::vec& d, int k);

double step_size_gss(const Variables& vars, const PartialLossConstants& consts,
                     const arma::mat& Rstar0_inv,
                     const arma::sp_mat& W_cpath,
                     const arma::mat& W_lasso,
                     const arma::vec& ddir, double lambda_cpath,
                     double lambda_lasso, double eps_lasso, int k, double lo,
                     double hi, double tol);
