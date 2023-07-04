#include <RcppEigen.h>


double squaredNormTheta(const Eigen::MatrixXd& Theta, int i, int j);

double normRA(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
              const Eigen::VectorXi& p, int i, int j);
