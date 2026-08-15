#ifndef UTILS_H
#define UTILS_H

#include <RcppArmadillo.h>


struct Variables;

arma::mat compute_Theta(const arma::mat& R, const arma::vec& A,
                        const arma::ivec& u);

double square(double x);

arma::sp_mat
convert_to_sparse(const arma::mat& W_keys,
                  const arma::vec& W_values, int n_variables);

arma::mat compute_R_star0_inv(const Variables& vars, int k);

arma::vec drop_variable(const arma::vec& x, int k);

void drop_variable_inplace(arma::vec& x, int k);

void drop_variable_inplace(arma::mat& X, int k);

void update_inverse_inplace(arma::mat& M_inv, const arma::mat& M,
                            int k);

double partial_trace(const arma::mat& S, const arma::ivec& u, int k);

double sum_selected_elements(const arma::mat& S, const arma::ivec& u,
                             const arma::ivec& p, int k);

arma::vec
sum_multiple_selected_elements(const arma::mat& S,
                               const arma::ivec& u,
                               const arma::ivec& p, int k);

void update_RA_inplace(arma::mat& R, arma::vec& A,
                       const arma::vec& values, int k);

#endif // UTILS_H
