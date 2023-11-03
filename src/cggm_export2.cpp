#include <RcppEigen.h>
#include <iostream>
#include "gradient2.h"
#include "hessian2.h"
#include "loss2.h"
#include "utils2.h"
#include "variables.h"


Eigen::SparseMatrix<double>
fuseW(const Eigen::SparseMatrix<double>& W, const Eigen::VectorXi& u)
{
    /* Fuse rows/columns of the weight matrix based on a new membership vector
     *
     * Inputs:
     * W: old sparse weight matrix
     * u: membership vector, has length equal the the number of old clusters
     *
     * Output:
     * New sparse weight matrix
     */

    // Number of nnz elements
    int nnz = W.nonZeros();

    // Initialize list of triplets
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);

    // Fill list of triplets
    for (int j = 0; j < W.outerSize(); j++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(W, j); it; ++it) {
            // Row index
            int i = it.row();

            // New indices
            int ii = u[i];
            int jj = u[j];

            // If the value would land on the diagonal, continue to the next one
            if (ii == jj) {
                continue;
            }

            // Add to the triplets
            triplets.push_back(
                Eigen::Triplet<double>(ii, jj, it.value())
            );
        }
    }

    // Construct the sparse matrix
    int n_clusters = u.maxCoeff() + 1;
    Eigen::SparseMatrix<double> result(n_clusters, n_clusters);
    result.setFromTriplets(triplets.begin(), triplets.end());

    return result;
}


void NewtonDescent(Variables& vars, const Eigen::MatrixXd& S,
                   const Eigen::SparseMatrix<double>& W, double lambda, int k,
                   double gss_tol, int verbose)
{
    /* Compute Newton descent direction for variables relating to cluster k and
     * find a step size that decreases the loss function
     *
     * Could be written as a member function of the Variables struct
     *
     * Inputs:
     * vars: struct containing the optimization variables
     * S: sample covariance matrix
     * W: sparse weight matrix
     * lambda: regularization parameter
     * k: cluster of interest
     * gss_tol: tolerance for the golden section search
     * verbose: level of information printed to console
     *
     * Output:
     * New sparse weight matrix
     */

    // Compute the inverse of R*
    Eigen::MatrixXd Rstar0_inv = computeRStar0Inv2(vars, k);

    // Compute gradient
    Eigen::VectorXd g = gradient2(vars, Rstar0_inv, S, W, lambda, k);

    // Compute Hessian
    Eigen::MatrixXd H = hessian2(vars, Rstar0_inv, S, W, lambda, k);

    // Compute descent direction, experimentation with different solvers is
    // possible
    Eigen::VectorXd d = -H.ldlt().solve(g);

    Rcpp::Rcout << d << '\n';
}


// [[Rcpp::export]]
void test(const Eigen::MatrixXd& W_keys, const Eigen::VectorXd& W_values,
          const Eigen::MatrixXd& Ri, const Eigen::VectorXd& Ai,
          const Eigen::VectorXi& pi, const Eigen::VectorXi& ui,
          const Eigen::MatrixXd& S, const Eigen::VectorXd& lambdas)
{
    /* Inputs:
     * W_keys: indices for the nonzero elements of the weight matrix
     * W_values: nonzero elements of the weight matrix
     *
     */

    Rcpp::Rcout << std::fixed;
    Rcpp::Rcout.precision(5);

    auto W = convertToSparse(W_keys, W_values, S.cols());
    Rcpp::Rcout << W << '\n';

    //Rcpp::Rcout << W.col(1) << '\n';
    //Rcpp::Rcout << W.row(1) << '\n';

    Variables vars(Ri, Ai, W, pi, ui);

    Rcpp::Rcout << lossComplete(vars, S, W, lambdas(0)) << '\n';

    int k = 0;

    Eigen::MatrixXd Rstar0_inv = computeRStar0Inv2(vars, k);
    Rcpp::Rcout << Rstar0_inv << "\n\n";

    auto grad = gradient2(vars, Rstar0_inv, S, W, lambdas(0), k);
    Rcpp::Rcout << grad << "\n\n";

    auto hess = hessian2(vars, Rstar0_inv, S, W, lambdas(0), k);
    Rcpp::Rcout << hess << '\n';

    NewtonDescent(vars, S, W, lambdas(0), k, 1e-6, 0);
}
