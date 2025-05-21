#include <RcppEigen.h>
#include <chrono>
#include <list>
#include <string>
#include "gradient.h"
#include "hessian.h"
#include "loss.h"
#include "partial_loss_constants.h"
#include "result.h"
#include "step_size.h"
#include "utils.h"
#include "variables.h"
#include "clock.h"


Clock CLOCK;


Eigen::SparseMatrix<double> fuse_W(const Eigen::SparseMatrix<double>& W_cpath,
                                   const Eigen::VectorXi& u)
{
    /* Fuse rows/columns of the weight matrix based on a new membership vector
     *
     * Inputs:
     * W_cpath: old sparse weight matrix
     * u: membership vector, has length equal the the number of old clusters
     *
     * Output:
     * New sparse weight matrix
     */

    // Number of nnz elements
    int nnz = W_cpath.nonZeros();

    // Initialize list of triplets
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);

    // Fill list of triplets
    for (int j = 0; j < W_cpath.outerSize(); j++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(W_cpath, j); it; ++it) {
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


void Newton_descent(Variables& vars, const Eigen::MatrixXd& Rstar0_inv,
                    const Eigen::MatrixXd& S,
                    const Eigen::SparseMatrix<double>& W_cpath,
                    const Eigen::MatrixXd& W_lasso, double lambda_cpath,
                    double lambda_lasso, double eps_lasso, int k,
                    double gss_tol, bool refit,
                    const Eigen::MatrixXi& refit_lasso, int verbose)
{
    /* Compute Newton descent direction for variables relating to cluster k and
     * find a step size that decreases the loss function
     *
     * Inputs:
     * vars: struct containing the optimization variables
     * S: sample covariance matrix
     * W_cpath: sparse weight matrix
     * lambda_cpath: regularization parameter
     * k: cluster of interest
     * gss_tol: tolerance for the golden section search
     * direction should be used
     * verbose: level of information printed to console
     *
     * Output:
     * None, the optimization variables are modified in place
     */

    CLOCK.tick("Descent Direction");
    // Compute the inverse of R*
    // Eigen::MatrixXd Rstar0_inv = compute_R_star0_inv(vars, k);

    // Compute gradient
    CLOCK.tick("Descent Direction - Gradient");
    Eigen::VectorXd g = gradient(
        vars, Rstar0_inv, S, W_cpath, W_lasso, lambda_cpath, lambda_lasso,
        eps_lasso, k
    );
    CLOCK.tock("Descent Direction - Gradient");

    // Compute descent direction
    Eigen::VectorXd d;

    // Compute Hessian
    CLOCK.tick("Descent Direction - Hessian");
    Eigen::MatrixXd H = hessian(
        vars, Rstar0_inv, S, W_cpath, W_lasso, lambda_cpath, lambda_lasso,
        eps_lasso, k
    );
    CLOCK.tock("Descent Direction - Hessian");

    // Solve for descent direction
    CLOCK.tick("Descent Direction - Compute H^-1 g");
    if (H.cols() <= 20) {
        // Slower, more accurate solver for small Hessian
        d = -H.colPivHouseholderQr().solve(g);
    } else {
        // Faster, less accurate solver for larger Hessian
        d = -H.ldlt().solve(g);
    }
    CLOCK.tock("Descent Direction - Compute H^-1 g");

    // Check if a refitting procedure is happening
    if (refit) {
        for (int l = 0; l < refit_lasso.cols(); l++) {
            // If the element of R should not be changed, set its descent
            // direction to zero
            if (refit_lasso(l, k) == 0) {
                d(1 + l) = 0;
            }
        }
    }
    CLOCK.tock("Descent Direction");

    CLOCK.tick("Step Size Selection");
    // Compute interval for allowable step sizes
    Eigen::VectorXd step_sizes = max_step_size(vars, Rstar0_inv, d, k);

    // Set minimum step size to 0. Maximum could be set to a lower value (i.e.,
    // 2) to improve computation times, but may lead to undesired side effects
    step_sizes(0) = 0.0;
    step_sizes(1) = std::min(2.0, step_sizes(1));

    // Precompute constants that are used in the loss for cluster k
    PartialLossConstants consts(vars, S, k);

    // Find the optimal step size
    double s = step_size_gss(
        vars, consts, Rstar0_inv, S, W_cpath, W_lasso, d, lambda_cpath,
        lambda_lasso, eps_lasso, k, step_sizes(0), step_sizes(1), gss_tol
    );
    CLOCK.tock("Step Size Selection");

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


void fuse_clusters(Variables& vars, Eigen::SparseMatrix<double>& W_cpath,
                   Eigen::MatrixXd& W_lasso, int k, int m)
{
    /* Perform fusion of clusters k and m
     *
     * Inputs:
     * vars: struct containing the optimization variables
     * W_cpath: sparse weight matrix
     * k: cluster of interest
     * m: cluster k is fused with
     *
     * Output:
     * None, the optimization variables and sparse weight matrix are modified
     * in place
     */

    // Current number of clusters
    int n_clusters = W_cpath.cols();

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

    // Fuse the clusterpath weight matrix
    W_cpath = fuse_W(W_cpath, u_new);

    // Fuse the lasso weight matrix
    W_lasso.col(k) += W_lasso.col(m);
    W_lasso.row(k) += W_lasso.row(m);
    drop_variable_inplace(W_lasso, m);

    // Fuse the optimization variables
    vars.fuse_clusters(k, m, W_cpath);
}


// [[Rcpp::export(.cggm)]]
Rcpp::List cggm(const Eigen::MatrixXd& W_keys, const Eigen::VectorXd& W_values,
                const Eigen::MatrixXd& W_lassoi, const Eigen::MatrixXd& Ri,
                const Eigen::VectorXd& Ai, const Eigen::VectorXi& pi,
                const Eigen::VectorXi& ui, const Eigen::MatrixXd& S,
                const Eigen::VectorXd& lambdas, double lambda_lasso,
                double eps_lasso, double eps_fusions, double scale_factor_cpath,
                double scale_factor_lasso, double gss_tol, double conv_tol,
                int max_iter, bool store_all_res, bool refit,
                const Eigen::MatrixXi& refit_lasso, int verbose)
{
    /* Inputs:
     * W_keys: indices for the nonzero elements of the weight matrix
     * W_values: nonzero elements of the weight matrix
     *
     */
    // Register start time for tracking the duration of the iterations
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();

    // Scale the lasso penalty parameter
    lambda_lasso *= scale_factor_lasso;

    CLOCK.tick("CGGM");

    // Printing settings
    Rcpp::Rcout << std::fixed;
    Rcpp::Rcout.precision(5);

    // Construct the sparse weight matrix
    auto W_cpath = convert_to_sparse(W_keys, W_values, Ri.cols());

    // Copy the lasso weight matrix
    Eigen::MatrixXd W_lasso(W_lassoi);

    // Linked list with results
    LinkedList results;

    // Store minimization results
    std::list<Eigen::VectorXd> loss_progressions;
    std::list<Eigen::VectorXd> loss_timings;

    // Struct with optimization variables
    Variables vars(Ri, Ai, W_cpath, pi, ui);

    // Minimize  for each value for lambda_cpath
    for (int lambda_index = 0; lambda_index < lambdas.size(); lambda_index++) {
        // Clusterpath lambda
        double lambda_cpath = lambdas(lambda_index) * scale_factor_cpath;

        // Current value of the loss and "previous" value
        double l1 = loss_complete(
            vars, S, W_cpath, W_lasso, lambda_cpath, lambda_lasso, eps_lasso
        );
        double l0 = 1.0 + 2 * l1;

        // Vector of loss function values
        Eigen::VectorXd loss_values(max_iter + 1);
        loss_values(0) = l1;

        // Vector of timings for loss function values
        Eigen::VectorXd loss_timestamps(max_iter + 1);
        loss_timestamps(0) = 0.0;

        // Starting time
        std::chrono::high_resolution_clock::time_point start_time =
            std::chrono::high_resolution_clock::now();

        // Iteration counter
        int iter = 0;

        // Initialize the inverse of R*
        CLOCK.tick("Computation of inverse of R*0");
        Eigen::MatrixXd Rstar0_inv = compute_R_star0_inv(vars, 0);
        CLOCK.tock("Computation of inverse of R*0");

        // Flag to indicate that Rstar0_inv should be updated
        bool update_Rstar0_inv = false;

        while((l0 - l1) / l0 > conv_tol && iter < max_iter) {
            // Keep track of whether a fusion occurred
            bool fused = false;

            // While loop as the stopping criterion may change during the loop
            int k = 0;

            while (k < vars.m_R.cols()) {
                // Check if there is another cluster that k should fuse with,
                // but only if the clusterpath lambda is positive. The value -1
                // indicates no elligible fusions are found
                int fusion_index = -1;
                if (lambda_cpath > 0) {
                    fusion_index = fusion_check(vars, eps_fusions, k);
                }

                // If no fusion candidate is found, perform coordinate descent
                // with Newton descent direction
                if (fusion_index < 0) {
                    CLOCK.tick("Computation of inverse of R*0");
                    // If the number of clusters has changed, recompute the
                    // inverse of R* from scratch
                    if ((vars.m_R.cols() - 1) != Rstar0_inv.cols()) {
                        Rstar0_inv = compute_R_star0_inv(vars, k);
                    }
                    // Update the inverse of R*, this is not necessary if this
                    // is the first iteration of the minimization for the
                    // current value for lambda_cpath
                    else if (update_Rstar0_inv && Rstar0_inv.cols() > 0) {
                        update_inverse_inplace(Rstar0_inv, vars.m_Rstar, k);
                    }

                    // After the first iteration, Rstar0_inv should always be
                    // computed: via update or complete recomputation
                    update_Rstar0_inv = true;

                    CLOCK.tock("Computation of inverse of R*0");

                    // Perform Newton descent
                    Newton_descent(
                        vars, Rstar0_inv, S, W_cpath, W_lasso, lambda_cpath,
                        lambda_lasso, eps_lasso, k, gss_tol, refit, refit_lasso,
                        verbose
                    );

                    // Increment k
                    k++;
                }
                // Otherwise, perform a fusion of k and fusion_index
                else {
                    fuse_clusters(vars, W_cpath, W_lasso, k, fusion_index);

                    // If the removed cluster had an index smaller than k,
                    // decrement k
                    k -= (fusion_index < k);
                }
            }

            // At the end of the iteration, compute the new loss
            l0 = l1;
            l1 = loss_complete(
                vars, S, W_cpath, W_lasso, lambda_cpath, lambda_lasso, eps_lasso
            );

            // Increment iteration counter
            iter++;

            // Add loss function value
            loss_values(iter) = l1;

            // Record iteration end time
            std::chrono::high_resolution_clock::time_point end_time =
                std::chrono::high_resolution_clock::now();
            loss_timestamps(iter) =
                std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

            // If a fusion occurred, guarantee an extra iteration
            if (fused) {
                l0 = l1 / (1 - conv_tol) + 1.0;
            }
        }

        // Rcpp::Rcout << "Number of iterations: " << iter << '\n';

        // Add the results to the list
        if ((results.get_size() < 1) || store_all_res ||
                (results.last_clusters() > vars.m_R.cols())) {
            results.insert(
                CGGMResult(
                    vars.m_R, vars.m_A, vars.m_u, lambdas(lambda_index), l1
                )
            );
        }

        // Add loss function values to the list
        loss_values.conservativeResize(iter + 1);
        loss_progressions.push_back(loss_values);
        loss_timestamps.conservativeResize(iter + 1);
        loss_timings.push_back(loss_timestamps);
    }

    CLOCK.tock("CGGM");
    // CLOCK.print(); // UNCOMMENT for printing diagnostics
    CLOCK.reset();

    // Print the minimization time
    if (verbose > 1) {
        std::chrono::high_resolution_clock::time_point end =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        Rcpp::Rcout << "Duration: " << dur.count() << '\n';
    }

    // Construct results
    auto R_results = results.convert_to_RcppList();

    // Progression of loss function
    Rcpp::List list_loss_progressions;
    for (int i = 0; i < lambdas.size(); i++) {
        list_loss_progressions[std::to_string(i + 1)] = loss_progressions.front();
        loss_progressions.pop_front();
    }
    R_results["loss_progression"] = list_loss_progressions;

    // Timelines for the loss function progression
    Rcpp::List list_loss_timings;
    for (int i = 0; i < lambdas.size(); i++) {
        list_loss_timings[std::to_string(i + 1)] = loss_timings.front();
        loss_timings.pop_front();
    }
    R_results["loss_timings"] = list_loss_timings;

    return R_results;
}
