#include <RcppEigen.h>
#include "norms.h"
#include "partial_loss_constants.h"
#include "utils.h"
#include "variables.h"


double loss_complete(const Variables& vars, const Eigen::MatrixXd& S,
                     const Eigen::SparseMatrix<double>& W, double lambda)
{
    /* Compute the value of the entire loss function, including all variables
     *
     * Inputs:
     * vars: struct containing the optimization variables
     * S: sample covariance matrix
     * W: sparse weight matrix
     * lambda_cpath: regularization parameter
     *
     * Output:
     * The loss
     */

    // Create references to the variables in the struct
    const Eigen::MatrixXd &R = vars.m_R;
    const Eigen::MatrixXd &A = vars.m_A;
    const Eigen::VectorXi &p = vars.m_p;
    const Eigen::VectorXi &u = vars.m_u;
    const Eigen::SparseMatrix<double> &D = vars.m_D;

    // Number of clusters
    int n_clusters = R.cols();

    // Number of variables
    int n_variables = S.cols();

    // Compute log determinant
    Eigen::MatrixXd Rstar = vars.m_Rstar;
    for (int i = 0; i < R.cols(); i++) {
        Rstar.row(i) *= std::sqrt((double) p(i));
        Rstar.col(i) *= std::sqrt((double) p(i));
    }
    double loss_det = std::log(Rstar.determinant());

    for (int i = 0; i < n_clusters; i++) {
        loss_det += (p(i) - 1) * std::log(A(i) - R(i, i));
    }

    // Covariance part of the loss
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
    if (lambda <= 0) {
        return -loss_det + loss_cov;
    }

    // Clusterpath part
    double loss_cpath = 0;

    for (int i = 0; i < W.outerSize(); i++) {
        // Iterators
        Eigen::SparseMatrix<double>::InnerIterator D_it(D, i);
        Eigen::SparseMatrix<double>::InnerIterator W_it(W, i);

        for (; W_it; ++W_it) {
            if (W_it.col() > W_it.row()) {
                loss_cpath += W_it.value() * D_it.value();
            }

            // Continue iterator for D
            ++D_it;
        }
    }

    return -loss_det + loss_cov + lambda * loss_cpath;
}


double loss_partial(const Variables vars, const PartialLossConstants& consts,
                    const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
                    const Eigen::MatrixXd& Rstar0_inv, const Eigen::MatrixXd& S,
                    const Eigen::SparseMatrix<double>& W, double lambda, int k)
{
    // Create references to the variables in the structs
    const Eigen::VectorXi &p = vars.m_p;
    const Eigen::VectorXi &u = vars.m_u;
    const Eigen::SparseMatrix<double> &E = consts.m_E;

    // Number of rows/columns in R
    int n_clusters = R.cols();

    // Parts of the update
    double r_kk = R(k, k);
    double a_kk = A(k);
    Eigen::VectorXd r_k = R.row(k);
    drop_variable_inplace(r_k, k);

    // Determinant part
    double loss_det = (p(k) - 1) * r_kk - p(k) * r_k.dot(Rstar0_inv * r_k);
    loss_det = std::log(a_kk + loss_det) + (p(k) - 1) * std::log(a_kk - r_kk);

    //  Covariance part
    double loss_cov = 2 * r_k.dot(consts.m_uSU) + consts.m_uSu * r_kk;
    loss_cov += (a_kk - r_kk) * consts.m_pTraceS;

    // Return if lambda is not positive
    if (lambda <= 0) {
        return -loss_det + loss_cov;
    }

    // Clusterpath part
    double loss_cpath = 0;

    for (int j = 0; j < W.outerSize(); j++) {
        // Iterators
        Eigen::SparseMatrix<double>::InnerIterator E_it(E, j);
        Eigen::SparseMatrix<double>::InnerIterator W_it(W, j);

        for (; W_it; ++W_it) {
            // Index
            int i = W_it.row();

            // Skip loop for half of the computations
            if (i <= j) {
                ++E_it;
                continue;
            }

            // If i and j are not equal to k, there is a more efficient approach
            // to computing the loss
            if (i == k || j == k) {
                loss_cpath += W_it.value() * norm_RA(R, A, p, i, j);
            } else {
                // Compute distance
                double d_ij = E_it.value() + p(k) * square(R(i, k) - R(j, k));
                d_ij = std::sqrt(d_ij);

                // Add to the loss
                loss_cpath += W_it.value() * d_ij;
            }

            // Continue iterator for D
            ++E_it;
        }
    }

    return -loss_det + loss_cov + lambda * loss_cpath;
}
