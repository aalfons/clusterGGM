#include <RcppArmadillo.h>
#include <list>
#include <map>
#include <string>
#include <vector>
#include "gradient.h"
#include "hessian.h"
#include "loss.h"
#include "partial_loss_constants.h"
#include "result.h"
#include "step_size.h"
#include "utils.h"
#include "variables.h"


arma::sp_mat fuse_W(const arma::sp_mat& W_cpath,
                    const arma::ivec& u)
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

    std::map<std::pair<arma::uword, arma::uword>, double> entries;

    // Fill the map
    for (int j = 0; j < (int) W_cpath.n_cols; j++) {
        for (auto it = W_cpath.begin_col(j); it != W_cpath.end_col(j); ++it) {
            // Row index
            int i = it.row();

            // New indices
            int ii = u[i];
            int jj = u[j];

            // If the value would land on the diagonal, continue to the next one
            if (ii == jj) {
                continue;
            }

            // Add to the map
            entries[{(arma::uword) ii, (arma::uword) jj}] += *it;
        }
    }

    // Construct the sparse matrix
    int n_clusters = u.max() + 1;

    arma::umat locations(2, entries.size());
    arma::vec values(entries.size());
    arma::uword idx = 0;
    for (const auto& entry : entries) {
        locations(0, idx) = entry.first.first;
        locations(1, idx) = entry.first.second;
        values(idx) = entry.second;
        idx++;
    }

    arma::sp_mat result(locations, values, n_clusters, n_clusters);

    return result;
}


void Newton_descent(Variables& vars, const arma::mat& Rstar0_inv,
                    const arma::mat& S,
                    const arma::sp_mat& W_cpath,
                    const arma::mat& W_lasso, double lambda_cpath,
                    double lambda_lasso, double eps_lasso, int k,
                    double gss_tol, bool refit,
                    const arma::imat& refit_lasso, int verbose)
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
    // Compute gradient
    arma::vec g = gradient(
        vars, Rstar0_inv, S, W_cpath, W_lasso, lambda_cpath, lambda_lasso,
        eps_lasso, k
    );

    // Compute descent direction
    arma::vec d;

    // Compute Hessian
    arma::mat H = hessian(
        vars, Rstar0_inv, S, W_cpath, W_lasso, lambda_cpath, lambda_lasso,
        eps_lasso, k
    );

    // Solve for descent direction
    if (H.n_cols <= 20) {
        // Slower, more accurate solver for small Hessian
        d = -arma::solve(H, g);
    } else {
        // Faster solver for larger Hessian, exploiting that it is symmetric
        d = -arma::solve(H, g, arma::solve_opts::likely_sympd);
    }

    // Check if a refitting procedure is happening
    if (refit) {
        for (int l = 0; l < (int) refit_lasso.n_cols; l++) {
            // If the element of R should not be changed, set its descent
            // direction to zero
            if (refit_lasso(l, k) == 0) {
                d(1 + l) = 0;
            }
        }
    }

    // Compute interval for allowable step sizes
    arma::vec step_sizes = max_step_size(vars, Rstar0_inv, d, k);

    // Set minimum step size to 0. Maximum could be set to a lower value (i.e.,
    // 2) to improve computation times, but may lead to undesired side effects
    step_sizes(0) = 0.0;
    step_sizes(1) = std::min(2.0, step_sizes(1));

    // Precompute constants that are used in the loss for cluster k
    PartialLossConstants consts(vars, S, k);

    // Find the optimal step size
    double s = step_size_gss(
        vars, consts, Rstar0_inv, W_cpath, W_lasso, d, lambda_cpath,
        lambda_lasso, eps_lasso, k, step_sizes(0), step_sizes(1), gss_tol
    );

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

    // Get minimum value
    arma::uword e = vars.m_W.col_ptrs[k];

    for (auto W_it = vars.m_W.begin_col(k); W_it != vars.m_W.end_col(k); ++W_it, ++e) {
        if (min_val > vars.m_D(e)) {
            min_val = vars.m_D(e);
            min_idx = W_it.row();
        }
    }

    // Check if the minimum distance is smaller than the threshold, if so, it
    // is an eligible fusion
    if (min_val <= eps_fusions) {
        return min_idx;
    }

    return -1;
}


void fuse_clusters(Variables& vars, arma::sp_mat& W_cpath,
                   arma::mat& W_lasso, int k, int m)
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
    int n_clusters = W_cpath.n_cols;

    // Membership vector that translates the current clusters to the new
    // situation
    arma::ivec u_new(n_clusters);

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
Rcpp::List cggm(const arma::mat& W_keys, const arma::vec& W_values,
                const arma::mat& W_lassoi, const arma::mat& Ri,
                const arma::vec& Ai, const arma::ivec& pi,
                const arma::ivec& ui, const arma::mat& S,
                const arma::vec& lambdas, double lambda_lasso,
                double eps_lasso, double eps_fusions, double scale_factor_cpath,
                double scale_factor_lasso, double gss_tol, double conv_tol,
                int max_iter, bool store_all_res, bool refit,
                const arma::imat& refit_lasso, int verbose)
{
    /* Inputs:
     * W_keys: indices for the nonzero elements of the weight matrix
     * W_values: nonzero elements of the weight matrix
     *
     */
    // Scale the lasso penalty parameter
    lambda_lasso *= scale_factor_lasso;

    // Printing settings
    Rcpp::Rcout << std::fixed;
    Rcpp::Rcout.precision(5);

    // Construct the sparse weight matrix
    auto W_cpath = convert_to_sparse(W_keys, W_values, Ri.n_cols);

    // Copy the lasso weight matrix
    arma::mat W_lasso(W_lassoi);

    // Linked list with results
    std::vector<CGGMResult> results;

    // Store minimization loss function values.
    std::list<Rcpp::NumericVector> loss_progressions;

    // Struct with optimization variables
    Variables vars(Ri, Ai, W_cpath, pi, ui);

    // Minimize  for each value for lambda_cpath
    for (int lambda_index = 0; lambda_index < (int) lambdas.n_elem; lambda_index++) {
        // Clusterpath lambda
        double lambda_cpath = lambdas(lambda_index) * scale_factor_cpath;

        // Current value of the loss and "previous" value
        double l1 = loss_complete(
            vars, S, W_cpath, W_lasso, lambda_cpath, lambda_lasso, eps_lasso
        );
        double l0 = 1.0 + 2 * l1;

        // Vector of loss function values
        arma::vec loss_values(max_iter + 1);
        loss_values(0) = l1;

        // Iteration counter
        int iter = 0;

        // Initialize the inverse of R*
        arma::mat Rstar0_inv = compute_R_star0_inv(vars, 0);

        // Flag to indicate that Rstar0_inv should be updated
        bool update_Rstar0_inv = false;

        while((l0 - l1) / l0 > conv_tol && iter < max_iter) {
            // Keep track of whether a fusion occurred
            bool fused = false;

            // While loop as the stopping criterion may change during the loop
            int k = 0;

            while (k < (int) vars.m_R.n_cols) {
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
                    // If the number of clusters has changed, recompute the
                    // inverse of R* from scratch
                    if (((int) vars.m_R.n_cols - 1) != (int) Rstar0_inv.n_cols) {
                        Rstar0_inv = compute_R_star0_inv(vars, k);
                    }
                    // Update the inverse of R*, this is not necessary if this
                    // is the first iteration of the minimization for the
                    // current value for lambda_cpath
                    else if (update_Rstar0_inv && Rstar0_inv.n_cols > 0) {
                        update_inverse_inplace(Rstar0_inv, vars.m_Rstar, k);
                    }

                    // After the first iteration, Rstar0_inv should always be
                    // computed: via update or complete recomputation
                    update_Rstar0_inv = true;

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
                    fused = true;

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

            // If a fusion occurred, guarantee an extra iteration
            if (fused) {
                l0 = l1 / (1 - conv_tol) + 1.0;
            }
        }

        // Add the results to the list
        if (results.empty() || store_all_res ||
                (results.back().n_clusters > (int) vars.m_R.n_cols)) {
            results.emplace_back(
                vars.m_R, vars.m_A, vars.m_u, lambdas(lambda_index), l1
            );
        }

        // Add loss function values to the list
        loss_progressions.push_back(
            Rcpp::NumericVector(loss_values.begin(), loss_values.begin() + iter + 1)
        );
    }

    // Construct results
    auto R_results = convert_to_RcppList(results);

    // Progression of loss function
    Rcpp::List list_loss_progressions;
    for (int i = 0; i < (int) lambdas.n_elem; i++) {
        list_loss_progressions[std::to_string(i + 1)] = loss_progressions.front();
        loss_progressions.pop_front();
    }
    R_results["loss_progression"] = list_loss_progressions;

    return R_results;
}
