#include <RcppEigen.h>


Eigen::VectorXd
dropVariable(const Eigen::VectorXd& x, int k);

void
dropVariableInplace(Eigen::VectorXd& x, int k);

void
dropVariableInplace(Eigen::MatrixXd& X, int k);

void
printVector(const Eigen::VectorXd& vec);

void
printVector(const Eigen::VectorXi& vec);

void
printMatrix(const Eigen::MatrixXd& mat);

std::pair<Eigen::MatrixXd, Eigen::VectorXd>
updateRA(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
         const Eigen::VectorXd& values, int k);

void
updateRAInplace(Eigen::MatrixXd& R, Eigen::VectorXd& A,
                const Eigen::VectorXd& values, int k);

Eigen::MatrixXd
updateInverse(const Eigen::MatrixXd& inverse, const Eigen::VectorXd& vec, int i);

double
partialTrace(const Eigen::MatrixXd& S, const Eigen::VectorXi& u, int k);

double
sumSelectedElements(const Eigen::MatrixXd& S, const Eigen::VectorXi& u,
                    const Eigen::VectorXi& p, int k);

Eigen::VectorXd
sumMultipleSelectedElements(const Eigen::MatrixXd& S,
                            const Eigen::VectorXi& u,
                            const Eigen::VectorXi& p, int k);

Eigen::MatrixXd
computeRStar0Inv(const Eigen::MatrixXd& R,
                 const Eigen::VectorXd& A,
                 const Eigen::VectorXi& p, int k);

Eigen::MatrixXd
computeTheta(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
             const Eigen::VectorXi& u);

void
setEqualToClusterInplace(Eigen::MatrixXd& R, Eigen::VectorXd& A,
                         const Eigen::VectorXi& p, int k, int m);

void
setEqualToClusterMeansInplace(Eigen::MatrixXd& R, Eigen::VectorXd& A,
                              const Eigen::VectorXi& p, int k, int m);

double square(double x);
