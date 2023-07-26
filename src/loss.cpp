#include <RcppEigen.h>
#include <set>
#include "norms.h"
#include "utils.h"
#include "clock.h"


// [[Rcpp::export]]
double lossRAk(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
               const Eigen::VectorXi& p, const Eigen::VectorXi& u,
               const Eigen::MatrixXd& R_star_0_inv,
               const Eigen::MatrixXd& S, const Eigen::MatrixXd& UWU,
               double lambda_cpath, int k)
{
    CLOCK.tick("misc - lossRAk excl cpath");

    // Number of rows/columns in R
    int n_clusters = R.cols();

    // Determinant part
    Eigen::VectorXd r_k = R.row(k);
    r_k = dropVariable(r_k, k);

    double loss_det = A(k) + (p(k) - 1) * R(k, k) - p(k) * r_k.dot(R_star_0_inv * r_k);
    loss_det = std::log(loss_det) + (p(k) - 1) * std::log(A(k) - R(k, k));

    // Covariance part
    double loss_cov = 2 * r_k.dot(sumMultipleSelectedElements(S, u, p, k));
    loss_cov += sumSelectedElements(S, u, p, k) * R(k, k);
    loss_cov += (A(k) - R(k, k)) * partialTrace(S, u, k);

    // Return if lambda is not positive
    if (lambda_cpath <= 0) {
        return -loss_det + loss_cov;
    }

    CLOCK.tock("misc - lossRAk excl cpath");

    CLOCK.tick("misc - lossRAk cpath");

    // Clusterpath part
    double loss_cpath = 0;

    for (int i = 0; i < n_clusters; i++) {
        for (int j = 0; j < i; j++) {
            if (UWU(j, i) != 0) loss_cpath += UWU(j, i) * normRA(R, A, p, j, i);
        }
    }

    CLOCK.tock("misc - lossRAk cpath");

    return -loss_det + loss_cov + lambda_cpath * loss_cpath;
}


// [[Rcpp::export]]
double lossRA(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
              const Eigen::VectorXi& p, const Eigen::VectorXi& u,
              const Eigen::MatrixXd& S, const Eigen::MatrixXd& UWU,
              double lambda_cpath)
{
    // Number of rows/columns in R
    int n_clusters = R.cols();
    int n_variables = S.cols();

    // Determinant part
    Eigen::MatrixXd R_star(R);

    for (int i = 0; i < n_clusters; i++) {
        R_star(i, i) += (A(i) - R(i, i)) / p(i);
    }

    for (int i = 0; i < n_clusters; i++) {
        R_star.row(i) *= std::sqrt((double) p(i));
        R_star.col(i) *= std::sqrt((double) p(i));
    }

    double loss_det = std::log(R_star.determinant());

    for (int i = 0; i < n_clusters; i++) {
        loss_det += (p(i) - 1) * std::log(A(i) - R(i, i));
    }

    // Covariance part
    double loss_cov = 0;

    for (int j = 0; j < n_variables; j++) {
        for (int i = 0; i < n_variables; i++) {
            // The computation of the relevant elements for tr(SURU)
            loss_cov += S(i, j) * R(u(i), u(j));

            // The part that concerns the diagonal A
            if (i == j) {
                loss_cov += (A(u(j)) - R(u(i), u(j))) * S(i, j);
            }
        }
    }

    // Return if lambda is not positive
    if (lambda_cpath <= 0) {
        return -loss_det + loss_cov;
    }

    // Clusterpath part
    double loss_cpath = 0;

    for (int i = 0; i < n_clusters; i++) {
        for (int j = 0; j < i; j++) {
            if (UWU(j, i) != 0) loss_cpath += UWU(j, i) * normRA(R, A, p, j, i);
        }
    }

    return -loss_det + loss_cov + lambda_cpath * loss_cpath;
}
