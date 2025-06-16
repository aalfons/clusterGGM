#ifndef UTILS_H
#define UTILS_H

#include <RcppEigen.h>


struct Variables;

double square(double x);

Eigen::SparseMatrix<double>
convert_to_sparse(const Eigen::MatrixXd& W_keys,
                  const Eigen::VectorXd& W_values, int n_variables);

Eigen::MatrixXd compute_R_star0_inv(const Variables& vars, int k);

Eigen::VectorXd drop_variable(const Eigen::VectorXd& x, int k);

void drop_variable_inplace(Eigen::VectorXd& x, int k);

void drop_variable_inplace(Eigen::MatrixXd& X, int k);

void update_inverse_inplace(Eigen::MatrixXd& M_inv, const Eigen::MatrixXd& M,
                            int k);

double partial_trace(const Eigen::MatrixXd& S, const Eigen::VectorXi& u, int k);

double sum_selected_elements(const Eigen::MatrixXd& S, const Eigen::VectorXi& u,
                             const Eigen::VectorXi& p, int k);

Eigen::VectorXd
sum_multiple_selected_elements(const Eigen::MatrixXd& S,
                               const Eigen::VectorXi& u,
                               const Eigen::VectorXi& p, int k);

std::pair<Eigen::MatrixXd, Eigen::VectorXd>
update_RA(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
          const Eigen::VectorXd& values, int k);

void update_RA_inplace(Eigen::MatrixXd& R, Eigen::VectorXd& A,
                       const Eigen::VectorXd& values, int k);

#endif // UTILS_H
