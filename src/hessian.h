#include <RcppEigen.h>


Eigen::MatrixXd hessian(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
                        const Eigen::VectorXi& p, const Eigen::VectorXi& u,
                        const Eigen::MatrixXd& R_star_0_inv,
                        const Eigen::MatrixXd& S, const Eigen::MatrixXd& UWU,
                        double lambda, int k);
