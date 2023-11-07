#include <RcppEigen.h>
#include <iostream>
#include "gradient2.h"
#include "hessian2.h"
#include "loss2.h"
#include "partial_loss_constants.h"
#include "step_size2.h"
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

    // Compute interval for allowable step sizes
    Eigen::VectorXd step_sizes = maxStepSize2(vars, Rstar0_inv, d, k);

    // Change  later, for now these lines are the same as the other version
    step_sizes(0) = 0;
    step_sizes(1) = std::min(step_sizes(1), 2.0);

    // Precompute constants that are used in the loss for cluster k
    PartialLossConstants consts(vars, S, k);

    // Find the optimal step size
    double s = stepSizeSelection(vars, consts, Rstar0_inv, S, W, d, lambda, k,
                                 step_sizes(0), step_sizes(1), gss_tol);

    // Compute the loss for the old situation
    double loss_old = lossComplete(vars, S, W, lambda);

    // Update R and A using the obtained step size, also, reuse the constant
    // parts of the distances
    vars.updateCluster(s * d, consts.m_E, k);

    // Compute the loss for the new situation
    double loss_new = lossComplete(vars, S, W, lambda);
    Rcpp::Rcout << "Step size: " << s << '\n';
    Rcpp::Rcout << "old loss:  " << loss_old << '\n';
    Rcpp::Rcout << "new loss:  " << loss_new << '\n';
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

    // Constants that should be inputs at some point
    int max_iter = 1;
    double conv_tol = 1e-6;

    // Printing settings
    Rcpp::Rcout << std::fixed;
    Rcpp::Rcout.precision(5);

    // Construct the sparse weight matrix
    auto W = convertToSparse(W_keys, W_values, Ri.cols());

    // Struct with optimization variables
    Variables vars(Ri, Ai, W, pi, ui);

    // Minimize  for each value for lambda
    for (int lambda_index = 0; lambda_index < lambdas.size(); lambda_index++) {
        // Current value of the loss and "previous" value
        double l1 = lossComplete(vars, S, W, lambdas(lambda_index));
        double l0 = 1.0 + 2 * l1;

        // Iteration counter
        int iter = 0;

        while((l0 - l1) / l0 > conv_tol && iter < max_iter) {
            // While loop as the stopping criterion may change during the loop
            int k = 0;

            while (k < vars.m_R.cols()) {
                // Perform coordinate descent with Newton descent direction
                NewtonDescent(vars, S, W, lambdas(lambda_index), k, 1e-6, 0);

                // Increment k
                k++;
            }

            // Increment iteration counter
            iter++;
        }
    }
}
