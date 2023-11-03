#ifndef LOSS_H
#define LOSS_H

#include <RcppEigen.h>
#include "variables.h"


double lossComplete(const Variables& vars, const Eigen::MatrixXd& S,
                    const Eigen::SparseMatrix<double>& W, double lambda_cpath)
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
    if (lambda_cpath <= 0) {
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

    return -loss_det + loss_cov + lambda_cpath * loss_cpath;
}

#endif // LOSS_H
