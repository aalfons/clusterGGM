#include <RcppEigen.h>
#include "utils.h"
#include "gradient.h"
#include "step_size.h"
#include "loss.h"
#include "norms.h"
#include "result.h"
#include "clock.h"
#include "hessian.h"


Clock CLOCK;


/* Function that performs gradient descent for row/column k and updates R and A
 * in place.
 *
 * Inputs:
 * R: R
 * A: A
 * p: cluster size vector
 * u: cluster membership vector
 * R_star_0_inv: inverse of R* excluding row/column k
 * S: sample covariance
 * UWU: result of U^T*W*U
 * lambda_cpath: regularization parameter
 * k: row/column of interest
 * gss_tol: tolerance for the golden section search
 * verbose: level of verbosity (for verbose > 1 additional computations are
 *          done, which makes the algorithm slower)
 */
void gradientDescent(Eigen::MatrixXd& R, Eigen::VectorXd& A,
                     const Eigen::VectorXi& p, const Eigen::VectorXi& u,
                     const Eigen::MatrixXd& R_star_0_inv,
                     const Eigen::MatrixXd& S, const Eigen::MatrixXd& UWU,
                     double lambda_cpath, int k, double gss_tol, bool Newton_dd,
                     int verbose)
{
    // Compute gradient
    CLOCK.tick("cggm - gradientDescent - gradient");
    Eigen::VectorXd grad = gradient(R, A, p, u, R_star_0_inv, S, UWU, lambda_cpath, k, -1);
    CLOCK.tock("cggm - gradientDescent - gradient");

    // REMOVE
    if (verbose > 5) Rcpp::Rcout << grad << "\n\n";

    if (Newton_dd) {
        CLOCK.tick("cggm - gradientDescent - hessian");
        Eigen::MatrixXd H = hessian(R, A, p, u, R_star_0_inv, S, UWU, lambda_cpath, k);
        CLOCK.tock("cggm - gradientDescent - hessian");

        // REMOVE
        if (verbose > 5) Rcpp::Rcout << H << "\n\n";

        // If the cluster size of k is one, set the corresponding diagonal element
        // to 1 to facilitate the inverse
        if (p(k) == 1) {
            H(k + 1, k + 1) = 1;
        }

        CLOCK.tick("cggm - gradientDescent - hessian");
        grad = H.inverse() * grad;
        CLOCK.tock("cggm - gradientDescent - hessian");
    }

    // REMOVE
    if (verbose > 5) Rcpp::Rcout << -grad << "\n\n";

    // Compute step size interval that keeps the solution in the domain
    CLOCK.tick("cggm - gradientDescent - maxStepSize");
    Eigen::VectorXd step_sizes = maxStepSize(R, A, p, R_star_0_inv, grad, k);
    CLOCK.tock("cggm - gradientDescent - maxStepSize");

    // REMOVE
    if (verbose > 5) Rcpp::Rcout << step_sizes << "\n\n";

    // Let the step size interval start at 0, because negative step sizes will
    // increase the loss function
    step_sizes(0) = 0.0;

    // If the Hessian is used, let the maximum step size be 2
    if (Newton_dd) {
        step_sizes(1) = std::min(step_sizes(1), 2.0);
    }

    // REMOVE
    if (verbose > 5) Rcpp::Rcout << step_sizes << "\n\n";

    // Compute optimal step size, let the minimum step size be 0 instead of
    // negative
    CLOCK.tick("cggm - gradientDescent - gssStepSize");
    double step_size = gssStepSize(
        R, A, p, u, R_star_0_inv, S, UWU, grad, lambda_cpath, k, step_sizes(0),
        step_sizes(1), gss_tol * step_sizes(1)
    );
    CLOCK.tock("cggm - gradientDescent - gssStepSize");

    // Declare variables possibly used for printing info
    double loss_old = 0;
    double loss_new = 0;

    // Compute the loss for the old situation
    if (verbose > 2) {
        loss_old = lossRA(R, A, p, u, S, UWU, lambda_cpath);
    }

    // Update R and A using the obtained step size
    updateRAInplace(R, A, -step_size * grad, k);

    // REMMOVE
    if (verbose > 5) Rcpp::Rcout << "\n" << R << "\n\n";
    if (verbose > 5) Rcpp::Rcout << "\n" << A << "\n\n";

    // Compute the loss for the new situation
    if (verbose > 2) {
        loss_new = lossRA(R, A, p, u, S, UWU, lambda_cpath);
    }

    // Print the step size found using golden section search
    if (verbose > 1) {
        Rcpp::Rcout << "gradient descent for row/column " << k + 1 << ":\n";
        Rcpp::Rcout << "    step size: " << step_size << '\n';

        // Print values for the loss function
        if (verbose > 2) {
            Rcpp::Rcout << "    old loss:  " << loss_old << '\n';
            Rcpp::Rcout << "    new loss:  " << loss_new << '\n';
        }
    }
}


Eigen::MatrixXd updateRStar0Inv(const Eigen::MatrixXd& R_star_0_inv,
                                const Eigen::MatrixXd& R,
                                const Eigen::MatrixXd& R_new,
                                const Eigen::VectorXd& A,
                                const Eigen::VectorXd& A_new,
                                const Eigen::VectorXi& p, int k, int m)
{
    // Vector that modifies R^* to turn into the clustered R^* if it is first
    // added to row m and then to column m, hence the division by two of the
    // mth element
    Eigen::VectorXd r_mod_2 = R_new.col(m) - R.col(m);
    r_mod_2(m) += (A_new(m) - R_new(m, m) - A(m) + R(m, m)) / p(m);
    r_mod_2(m) /= 2;
    dropVariableInplace(r_mod_2, k);

    // Compute the updated inverse
    Eigen::MatrixXd result = updateInverse(R_star_0_inv, r_mod_2, m - (m > k));

    return result;
}


bool fusionTestOneWay(const Eigen::MatrixXd& R_new, const Eigen::VectorXd& A_new,
                      const Eigen::VectorXi& p, const Eigen::VectorXi& u,
                      const Eigen::MatrixXd& R_star_0_inv,
                      const Eigen::MatrixXd& S, const Eigen::MatrixXd& UWU,
                      const Eigen::ArrayXd& G_inv, double lambda_cpath, int k,
                      int m, int verbose)
{
    // Compute gradient, this time there is a fusion candidate, which is indicated by m
    CLOCK.tick("cggm - fusionChecks - gradient");
    Eigen::VectorXd grad = gradient(R_new, A_new, p, u, R_star_0_inv, S, UWU, lambda_cpath, k, m);
    CLOCK.tock("cggm - fusionChecks - gradient");

    // Compute LHS and RHS of the test
    double test_lhs = std::sqrt((grad.array() * grad.array() * G_inv).sum());
    double test_rhs = lambda_cpath * UWU(k, m);

    // Perform the test
    if (verbose > 1) {
        Rcpp::Rcout << "test fusion of row/column " << k + 1 << " with " << m + 1 << ":\n";
        Rcpp::Rcout << "    " << test_lhs << " <= " << test_rhs;
        if (test_lhs <= test_rhs) Rcpp::Rcout << ": TRUE\n";
        if (test_lhs > test_rhs) Rcpp::Rcout << ": FALSE\n";
    }

    return test_lhs <= test_rhs;
}


bool fusionTestAsMean(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
                      const Eigen::MatrixXd& R_new, const Eigen::VectorXd& A_new,
                      const Eigen::VectorXi& p, const Eigen::VectorXi& u,
                      const Eigen::MatrixXd& R_star_0_inv,
                      const Eigen::MatrixXd& S, const Eigen::MatrixXd& UWU,
                      const Eigen::ArrayXd& G_inv, double lambda_cpath, int k,
                      int m, int fusion_type, int verbose)
{
    // Compute the new inverse of R^*0
    CLOCK.tick("cggm - fusionChecks - updateRStar0Inv");
    Eigen::MatrixXd R_star_0_inv_new = updateRStar0Inv(R_star_0_inv, R, R_new, A, A_new, p, k, m);
    CLOCK.tock("cggm - fusionChecks - updateRStar0Inv");

    // Compute gradient, this time there is a fusion candidate, which is indicated by m
    CLOCK.tick("cggm - fusionChecks - gradient");
    Eigen::VectorXd grad_k = gradient(R_new, A_new, p, u, R_star_0_inv_new, S, UWU, lambda_cpath, k, m);
    CLOCK.tock("cggm - fusionChecks - gradient");

    // Compute LHS and RHS of the test for k
    double test_lhs_k = std::sqrt((grad_k.array() * grad_k.array() * G_inv).sum());
    double test_rhs = lambda_cpath * UWU(k, m);

    // Perform the test
    if (verbose > 1) {
        Rcpp::Rcout << "test fusion of row/column " << k + 1 << " with " << m + 1 << ":\n";
        Rcpp::Rcout << "    " << k + 1 << " -> " << k + 1 << " & " << m + 1;
        Rcpp::Rcout << ": " << test_lhs_k << " <= " << test_rhs;
        if (test_lhs_k <= test_rhs) Rcpp::Rcout << ": TRUE\n";
        if (test_lhs_k > test_rhs) Rcpp::Rcout << ": FALSE\n";
    }

    // If the test fails, we can exit immediately
    if (test_lhs_k > test_rhs) {
        return false;
    } else if (fusion_type == 1) {
        return true;
    }

    // Compute the inverse of R^*0 from the perspective of row/column m
    CLOCK.tick("cggm - fusionChecks - computeRStar0Inv");
    Eigen::MatrixXd R_star_0_inv_m = computeRStar0Inv(R_new, A_new, p, m);
    CLOCK.tock("cggm - fusionChecks - computeRStar0Inv");

    // Compute the gradient for the mth row/column
    CLOCK.tick("cggm - fusionChecks - gradient");
    Eigen::VectorXd grad_m = gradient(R_new, A_new, p, u, R_star_0_inv_m, S, UWU, lambda_cpath, m, k);
    CLOCK.tock("cggm - fusionChecks - gradient");

    // Compute lhs of the test for m
    double test_lhs_m = std::sqrt((grad_m.array() * grad_m.array() * G_inv).sum());

    // Perform the test
    if (verbose > 1) {
        Rcpp::Rcout << "    " << m + 1 << " -> " << k + 1 << " & " << m + 1;
        Rcpp::Rcout << ": " << test_lhs_m << " <= " << test_rhs;
        if (test_lhs_m <= test_rhs) Rcpp::Rcout << ": TRUE\n";
        if (test_lhs_m > test_rhs) Rcpp::Rcout << ": FALSE\n";
    }

    return test_lhs_m <= test_rhs;
}


int fusionChecks(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
                 const Eigen::VectorXi& p, const Eigen::VectorXi& u,
                 const Eigen::MatrixXd& R_star_0_inv,
                 const Eigen::MatrixXd& S, const Eigen::MatrixXd& UWU,
                 double lambda_cpath, int k, double fusion_check_threshold,
                 int fusion_type, int verbose)
{
    // If lambda is zero no need to check for fusions
    if (lambda_cpath <= 0) {
        return -1;
    }

    // Number of clusters
    int n_clusters = R.cols();
    int n_variables = S.cols();

    // If there is one cluster left, no need to check anything
    if (n_clusters == 1) return -1;

    // Create copies of R and A to modify
    Eigen::MatrixXd R_new(R);
    Eigen::VectorXd A_new(A);

    // Construct G
    Eigen::ArrayXd G(n_clusters + 1);
    G(0) = 1;
    for (int i = 0; i < n_clusters; i++) {
        G(1 + i) = p(i);
    }
    G(1 + k) -= 1;

    for (int m = 0; m < n_clusters; m++) {
        if (m == k || UWU(k, m) <= 0) continue;

        // Compute the dissimilarity between k and m
        double d_km = normRA(R, A, p, k, m);

        // If the distance is larger than some threshold, do not even check
        // whether a fusion is appropriate
        if (d_km > fusion_check_threshold) continue;

        // Modify the (m+1)th value of G
        G(1 + m) -= 1;

        // Take the "inverse" of G
        for (int i = 0; i < G.size(); i++) {
            if (G(i) == 0) continue;
            G(i) = 1.0 / G(i);
        }

        // Variable for the test result
        bool test = false;

        if (fusion_type == 0) {
            // Set row/column k to the values in m
            CLOCK.tick("cggm - fusionChecks - setEqualToClusterInplace");
            setEqualToClusterInplace(R_new, A_new, p, k, m);
            CLOCK.tock("cggm - fusionChecks - setEqualToClusterInplace");

            // Perform the test for fusions where k is set to m
            test = fusionTestOneWay(R_new, A_new, p, u, R_star_0_inv, S, UWU,
                                    G, lambda_cpath, k, m, verbose);
        } else if (fusion_type == 1 || fusion_type == 2) {
            // Test with setting k and m to the average of these rows/cols
            CLOCK.tick("cggm - fusionChecks - setEqualToClusterMeansInplace");
            setEqualToClusterMeansInplace(R_new, A_new, p, k, m);
            CLOCK.tock("cggm - fusionChecks - setEqualToClusterMeansInplace");

            // Perform the test for fusions as mean of k and m
            test = fusionTestAsMean(
                R, A, R_new, A_new, p, u, R_star_0_inv, S, UWU, G, lambda_cpath,
                k, m, fusion_type, verbose
            );
        }

        // If the test succeeds, return the other fusion candidate
        if (test) return m;

        // Undo changes for the next iteration
        // Changes for kth variable
        A_new(k) = A(k);
        R_new.row(k) = R.row(k);
        R_new.col(k) = R.col(k);
        if (p(m) < 2) R_new(m, m) = R(m, m);

        // Changes for mth variable
        if (fusion_type == 1 || fusion_type == 2) {
            A_new(m) = A(m);
            R_new.row(m) = R.row(m);
            R_new.col(m) = R.col(m);
        }

        // Invert G back and add 1 to the (m+1)th element
        for (int i = 0; i < G.size(); i++) {
            if (G(i) == 0) continue;
            G(i) = 1.0 / G(i);
        }
        G(1 + m) += 1;
    }

    // Result stores the index that k should fuse with, if no fusion should be
    // done, the result is -1
    return -1;
}


int fusionChecksNaive(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
                      const Eigen::VectorXi& p, const Eigen::MatrixXd& UWU,
                      double lambda_cpath, int k, double fusion_check_threshold,
                      int verbose)
{
    // If lambda is zero no need to check for fusions
    if (lambda_cpath <= 0) return -1;

    // Number of clusters
    int n_clusters = R.cols();
    int n_variables = p.sum();

    // If there is one cluster left, no need to check anything
    if (n_clusters == 1) return -1;

    // Right hand side of the test to fuse
    double test_rhs = fusion_check_threshold;

    // Initialize minimum distance found
    double d_min = -1.0;

    // The index of the row/column k should be fused with
    int result = -1;

    // Find the row/column that is closest to row/column k
    for (int m = 0; m < n_clusters; m++) {
        if (m == k || UWU(m, k) <= 0) continue;

        // Compute the dissimilarity between k and m
        double d_km = normRA(R, A, p, k, m);

        if (d_km < d_min || d_min < 0) {
            // Replace the minimum distance
            d_min = d_km;

            // Store index
            result = m;
        }
    }

    if (verbose > 1) {
        Rcpp::Rcout << "test fusion of row/column " << k + 1;
        Rcpp::Rcout << " with " << result + 1 << ":\n";
        Rcpp::Rcout << "    " << k + 1 << ", " << result + 1 << " -> ";
        Rcpp::Rcout << k + 1 << " & " << result + 1;
        Rcpp::Rcout << ": " << d_min << " <= " << test_rhs;
        if (d_min <= test_rhs) Rcpp::Rcout << ": TRUE\n";
        if (d_min > test_rhs) Rcpp::Rcout << ": FALSE\n";
    }

    // Return the appropriate result
    if (d_min <= test_rhs) return result;
    return -1;
}


void performFusion(Eigen::MatrixXd& R, Eigen::VectorXd& A, Eigen::VectorXi& p,
                   Eigen::VectorXi& u, Eigen::MatrixXd& UWU, int k, int target,
                   int verbose, const Eigen::MatrixXd& S, double lambda_cpath,
                   int fusion_type)
{
    // Preliminaries
    int n_variables = u.size();
    int n_clusters = p.size();

    // Declare variables possibly used for printing info
    double loss_old = 0;
    double loss_new = 0;

    // Compute the loss for the old situation
    if (verbose > 2) {
        loss_old = lossRA(R, A, p, u, S, UWU, lambda_cpath);
    }

    // Do the replacing
    for (int i = 0; i < n_variables; i++) {
        if (u(i) == k) {
            u(i) = target;
        }
    }

    // Decrease all values that are larger than b by 1
    for (int i = 0; i < n_variables; i++) {
        if (u(i) > k) {
            u(i) -= 1;
        }
    }

    // Fusion of rows/cols in UWU
    UWU(target, target) += UWU(k, k) + UWU(k, target) + UWU(target, k);

    for (int i = 0; i < n_clusters; i++) {
        if (i == k || i == target) continue;

        UWU(i, target) += UWU(i, k);
        UWU(target, i) += UWU(i, k);
    }

    // Drop row/column k
    dropVariableInplace(UWU, k);

    // Fusions in R and A
    if (p(target) <= 1) {
        R(target, target) = R(k, target);
    }

    // If the mean is used to fuse two variables, modify R and A
    if (fusion_type == 1 || fusion_type == 2 || fusion_type == 3) {
        setEqualToClusterMeansInplace(R, A, p, k, target);
    }
    dropVariableInplace(R, k);
    dropVariableInplace(A, k);

    // Increase cluster size of the target cluster
    p(target) += p(k);

    // Move cluster sizes of clusters with index larger than k one position to
    // the left
    for (int i = k; i < n_clusters - 1; i++) {
        p(i) = p(i + 1);
    }
    p.conservativeResize(n_clusters - 1);

    // Compute the loss for the new situation and print the values
    if (verbose > 2) {
        loss_new = lossRA(R, A, p, u, S, UWU, lambda_cpath);

        Rcpp::Rcout << "    old loss:  " << loss_old << '\n';
        Rcpp::Rcout << "    new loss:  " << loss_new << '\n';
    }
}


// [[Rcpp::export(.cggm)]]
Rcpp::List cggm(const Eigen::MatrixXd& Ri, const Eigen::VectorXd& Ai,
                const Eigen::VectorXi& pi, const Eigen::VectorXi& ui,
                const Eigen::MatrixXd& S, const Eigen::MatrixXd& UWUi,
                const Eigen::VectorXd& lambdas, double gss_tol, double conv_tol,
                double fusion_check_threshold, int max_iter, bool store_all_res,
                int fusion_type, bool Newton_dd, bool print_profile_report,
                int verbose)
{
    /* fusion_type:
     * 0: no change in the target, k is set equal to m
     * 1: check whether the loss wrt k is minimized if k and m are set to the weighted mean
     * 2: check whether the losses wrt k and m are minimized if k and m are set to the weighted mean
     * 3: proximity based fusions
     */
    CLOCK.tick("cggm");

    // Set precision
    if (verbose > 0) {
        Rcpp::Rcout << std::fixed;
        Rcpp::Rcout.precision(5);
    }

    // Linked list with results
    LinkedList results;

    // For now the inputs may be already clustered
    Eigen::MatrixXd R(Ri);
    Eigen::VectorXd A(Ai);
    Eigen::VectorXi p(pi);
    Eigen::VectorXi u(ui);
    Eigen::MatrixXd UWU(UWUi);

    for (int lambda_index = 0; lambda_index < lambdas.size(); lambda_index++) {
        // Current value of the loss and "previous" value
        double l1 = lossRA(R, A, p, u, S, UWU, lambdas(lambda_index));
        double l0 = 1.0 + 2 * l1;

        // Iteration counter
        int iter = 0;

        while((l0 - l1) / l0 > conv_tol && iter < max_iter) {
            if (verbose > 1) {
                Rcpp::Rcout << "___________Iteration " << iter + 1 << "___________\n";
            }

            // Check for user interrupt every five iterations
            if (iter % 5 == 0) R_CheckUserInterrupt();

            // Keep track of whether a fusion occurred
            bool fused = false;

            // While loop as the stopping criterion may change during the loop
            int k = 0;
            while (k < R.cols()) {
                // REMOVE
                if (verbose > 5 && k > 1) break;

                CLOCK.tick("cggm - computeRStar0Inv");
                Eigen::MatrixXd R_star_0_inv = computeRStar0Inv(R, A, p, k);
                CLOCK.tock("cggm - computeRStar0Inv");

                // REMOVE
                if (verbose > 5) Rcpp::Rcout << R_star_0_inv << '\n';

                // Check if there is an eligible fusion
                CLOCK.tick("cggm - fusionChecks");
                int fusion_index = -1;

                if (fusion_type != 3) {
                    int fusion_index = fusionChecks(
                        R, A, p, u, R_star_0_inv, S, UWU, lambdas(lambda_index),
                        k, fusion_check_threshold, fusion_type, verbose
                    );
                } else {
                    fusion_index = fusionChecksNaive(
                        R, A, p, UWU, lambdas(lambda_index), k,
                        fusion_check_threshold, verbose
                    );
                }
                CLOCK.tock("cggm - fusionChecks");

                // No eligible fusions, proceed to gradient descent
                if (fusion_index < 0) {
                    CLOCK.tick("cggm - gradientDescent");
                    gradientDescent(R, A, p, u, R_star_0_inv, S, UWU, lambdas(lambda_index), k, gss_tol, Newton_dd, verbose);
                    CLOCK.tock("cggm - gradientDescent");

                    k++;
                } else {
                    CLOCK.tick("cggm - performFusion");
                    performFusion(R, A, p, u, UWU, k, fusion_index, verbose, S, lambdas(lambda_index), fusion_type);
                    CLOCK.tock("cggm - performFusion");

                    fused = true;
                }
            }

            l0 = l1;
            CLOCK.tick("cggm - lossRA");
            l1 = lossRA(R, A, p, u, S, UWU, lambdas(lambda_index));
            CLOCK.tock("cggm - lossRA");
            iter++;

            if (verbose > 0) {
                Rcpp::Rcout << "lambda: " << lambdas(lambda_index);
                Rcpp::Rcout << " | iteration: " << iter << " | loss: " << l1;
                Rcpp::Rcout << " | clusters: " << R.cols() << '\n';

                if (verbose > 1) Rcpp::Rcout << '\n';
            }

            // If a fusion occurred, guarantee an extra iteration
            if (fused) {
                l0 = l1 / (1 - conv_tol) + 1.0;
            }
        }

        if (results.getSize() < 1) {
            results.insert(CGGMResult(R, A, u, lambdas(lambda_index), l1));
        } else if (store_all_res) {
            results.insert(CGGMResult(R, A, u, lambdas(lambda_index), l1));
        } else if (results.lastClusters() > R.cols()) {
            results.insert(CGGMResult(R, A, u, lambdas(lambda_index), l1));
        }
    }

    CLOCK.tock("cggm");
    if (print_profile_report) CLOCK.print();
    CLOCK.reset();

    return results.convertToRcppList();
}
