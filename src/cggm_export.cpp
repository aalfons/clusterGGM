#include <RcppEigen.h>
#include <chrono>
#include <list>
#include "gradient.h"
#include "hessian.h"
#include "loss.h"
#include "partial_loss_constants.h"
#include "result.h"
#include "step_size.h"
#include "utils.h"
#include "variables.h"


Eigen::SparseMatrix<double> fuse_W(const Eigen::SparseMatrix<double>& W,
                                   const Eigen::VectorXi& u)
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


void Newton_descent(Variables& vars, const Eigen::MatrixXd& S,
                    const Eigen::SparseMatrix<double>& W, double lambda, int k,
                    double gss_tol, int verbose)
{
    /* Compute Newton descent direction for variables relating to cluster k and
     * find a step size that decreases the loss function
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
     * None, the optimization variables are modified in place
     */

    // Compute the inverse of R*
    Eigen::MatrixXd Rstar0_inv = compute_R_star0_inv(vars, k);

    // Compute gradient
    Eigen::VectorXd g = gradient(vars, Rstar0_inv, S, W, lambda, k);

    // Compute Hessian
    Eigen::MatrixXd H = hessian(vars, Rstar0_inv, S, W, lambda, k);

    // Compute descent direction
    Eigen::VectorXd d;

    // Slower, more accurate solver for small Hessian
    if (H.cols() <= 20) {
        d = -H.colPivHouseholderQr().solve(g);
    }
    // Faster, less accurate solver for larger Hessian
    else {
        d = -H.ldlt().solve(g);
    }

    // Compute interval for allowable step sizes
    Eigen::VectorXd step_sizes = max_step_size(vars, Rstar0_inv, d, k);

    // Set minimum step size to 0. Maximum could be set to a lower value (i.e.,
    // 2) to improve computation times, but may lead to undesired side effects
    step_sizes(0) = 0;

    // Precompute constants that are used in the loss for cluster k
    PartialLossConstants consts(vars, S, k);

    // Find the optimal step size
    double s = step_size_selection(vars, consts, Rstar0_inv, S, W, d, lambda, k,
                                   step_sizes(0), step_sizes(1), gss_tol);

    // Compute the loss for the old situation
    double loss_old = loss_complete(vars, S, W, lambda);

    // Update R and A using the obtained step size, also, reuse the constant
    // parts of the distances
    vars.update_cluster(s * d, consts.m_E, k);
}


int fusion_check(const Variables& vars, double eps_fusions, int k)
{
    /* Check for eligible fusions for cluster k
     *
     * Inputs:
     * vars: struct containing the optimization variables
     * eps_fusions: threshold for fusing two clusters
     * k: cluster of interest
     *
     * Output:
     * Index of the eligible cluster or -1 if there is none
     */

    // Initialize index and value of the minimum distance, as long as the
    // initial value of min_val is larger than eps_fusions, there is no issue
    double min_val = 1.0 + eps_fusions * 2;
    int min_idx = 0;

    // Iterator
    Eigen::SparseMatrix<double>::InnerIterator D_it(vars.m_D, k);

    // Get minimum value
    for (; D_it; ++D_it) {
        if (min_val > D_it.value()) {
            min_val = D_it.value();
            min_idx = D_it.row();
        }
    }

    // Check if the minimum distance is smaller than the threshold, if so, it
    // is an eligible fusion
    if (min_val <= eps_fusions) {
        return min_idx;
    }

    return -1;
}


void fuse_clusters(Variables& vars, Eigen::SparseMatrix<double>& W, int k,
                   int m)
{
    /* Perform fusion of clusters k and m
     *
     * Inputs:
     * vars: struct containing the optimization variables
     * W: sparse weight matrix
     * k: cluster of interest
     * m: cluster k is fused with
     *
     * Output:
     * None, the optimization variables and sparse weight matrix are modified
     * in place
     */

    // Current number of clusters
    int n_clusters = W.cols();

    // Membership vector that translates the current clusters to the new
    // situation
    Eigen::VectorXi u_new(n_clusters);

    // Up to m - 1 the cluster IDs are standard
    for (int i = 0; i < m; i++) {
        u_new(i) = i;
    }

    // The cluster ID of cluster m is k or k - 1, depending on which index is
    // larger
    u_new(m) = k - (m < k);

    // The cluster IDs of clusters beyond m are reduced by one to compensate for
    // the reduction in the number of clusters
    for (int i = m + 1; i < n_clusters; i++) {
        u_new(i) = i - 1;
    }

    // Fuse the weight matrix
    W = fuse_W(W, u_new);

    // Fuse the optimization variables
    vars.fuse_clusters(k, m, W);
}


// [[Rcpp::export(.cggm)]]
Rcpp::List cggm(const Eigen::MatrixXd& W_keys, const Eigen::VectorXd& W_values,
                const Eigen::MatrixXd& Ri, const Eigen::VectorXd& Ai,
                const Eigen::VectorXi& pi, const Eigen::VectorXi& ui,
                const Eigen::MatrixXd& S, const Eigen::VectorXd& lambdas,
                double eps_fusions, double scale_factor, double gss_tol,
                double conv_tol, int max_iter, bool store_all_res, int verbose)
{
    /* Inputs:
     * W_keys: indices for the nonzero elements of the weight matrix
     * W_values: nonzero elements of the weight matrix
     *
     */
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();

    // Printing settings
    Rcpp::Rcout << std::fixed;
    Rcpp::Rcout.precision(5);

    // Construct the sparse weight matrix
    auto W = convert_to_sparse(W_keys, W_values, Ri.cols());

    // Linked list with results
    LinkedList results;

    // Struct with optimization variables
    Variables vars(Ri, Ai, W, pi, ui);

    // Minimize  for each value for lambda
    for (int lambda_index = 0; lambda_index < lambdas.size(); lambda_index++) {
        // Current value of the loss and "previous" value
        double l1 = loss_complete(
            vars, S, W, lambdas(lambda_index) * scale_factor
        );
        double l0 = 1.0 + 2 * l1;

        // Iteration counter
        int iter = 0;

        while((l0 - l1) / l0 > conv_tol && iter < max_iter) {
            // Keep track of whether a fusion occurred
            bool fused = false;

            // While loop as the stopping criterion may change during the loop
            int k = 0;

            while (k < vars.m_R.cols()) {
                // Check if there is another cluster that k should fuse with
                int fusion_index = fusion_check(vars, eps_fusions, k);

                // If no fusion candidate is found, perform coordinate descent
                // with Newton descent direction
                if (fusion_index < 0) {
                    Newton_descent(
                        vars, S, W, lambdas(lambda_index) * scale_factor, k,
                        gss_tol, verbose
                    );

                    // Increment k
                    k++;
                }
                // Otherwise, perform a fusion of k and fusion_index
                else {
                    fuse_clusters(vars, W, k, fusion_index);

                    // If the removed cluster had an index smaller than k,
                    // decrement k
                    k -= (fusion_index < k);
                }
            }

            // At the end of the iteration, compute the new loss
            l0 = l1;
            l1 = loss_complete(
                vars, S, W, lambdas(lambda_index) * scale_factor
            );

            // Increment iteration counter
            iter++;

            // If a fusion occurred, guarantee an extra iteration
            if (fused) {
                l0 = l1 / (1 - conv_tol) + 1.0;
            }
        }

        // Add the results to the list
        if ((results.get_size() < 1) || store_all_res ||
                (results.last_clusters() > vars.m_R.cols())) {
            results.insert(
                CGGMResult(vars.m_R, vars.m_A, vars.m_u, lambdas(lambda_index),
                           l1)
            );
        }
    }

    // Print the minimization time
    if (verbose > 0) {
        std::chrono::high_resolution_clock::time_point end =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        Rcpp::Rcout << "Duration: " << dur.count() << '\n';
    }

    return results.convert_to_RcppList();
}
