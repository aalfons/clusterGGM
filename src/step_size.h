#include <RcppEigen.h>


Eigen::VectorXd maxStepSize(const Eigen::MatrixXd& R,
                            const Eigen::VectorXd& A,
                            const Eigen::VectorXi& p,
                            const Eigen::MatrixXd& R_star_0_inv,
                            const Eigen::VectorXd& g, int k);

double gssStepSize(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
                   const Eigen::VectorXi& p, const Eigen::VectorXi& u,
                   const Eigen::MatrixXd& R_star_0_inv,
                   const Eigen::MatrixXd& S, const Eigen::MatrixXd& UWU,
                   const Eigen::VectorXd& g, double lambda_cpath, int k,
                   double a, double b, double tol);
