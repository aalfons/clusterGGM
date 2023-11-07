#ifndef UTILS_H
#define UTILS_H

#include <RcppEigen.h>


struct Variables;

double square2(double x);

Eigen::SparseMatrix<double>
convertToSparse(const Eigen::MatrixXd& W_keys, const Eigen::VectorXd& W_values,
                int n_variables);

Eigen::MatrixXd computeRStar0Inv2(const Variables& vars, int k);

Eigen::VectorXd dropVariable2(const Eigen::VectorXd& x, int k);

void dropVariableInplace2(Eigen::VectorXd& x, int k);

double partialTrace2(const Eigen::MatrixXd& S, const Eigen::VectorXi& u, int k);

double sumSelectedElements2(const Eigen::MatrixXd& S, const Eigen::VectorXi& u,
                            const Eigen::VectorXi& p, int k);

Eigen::VectorXd
sumMultipleSelectedElements2(const Eigen::MatrixXd& S, const Eigen::VectorXi& u,
                             const Eigen::VectorXi& p, int k);

std::pair<Eigen::MatrixXd, Eigen::VectorXd>
updateRA2(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
          const Eigen::VectorXd& values, int k);

void updateRAInplace2(Eigen::MatrixXd& R, Eigen::VectorXd& A,
                      const Eigen::VectorXd& values, int k);

#endif // UTILS_H
