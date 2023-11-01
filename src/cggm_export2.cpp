#include <RcppEigen.h>
#include <iostream>
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

    Rcpp::Rcout << W.col(1) << '\n';
    Rcpp::Rcout << W.row(1) << '\n';

    W = fuseW(W, ui);
    Rcpp::Rcout << W << '\n';

    Variables vars(Ri, Ai, W, pi, ui);

    Rcpp::Rcout << lossComplete(vars, S, W, lambdas(0)) << '\n';
}







