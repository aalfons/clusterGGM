#include <RcppEigen.h>
#include "utils.h"
#include "gradient.h"
#include "step_size.h"
#include "loss.h"
#include "norms.h"
#include "result.h"
#include "clock.h"


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
                     double lambda_cpath, int k, double gss_tol,
                     int verbose)
{
    // Compute gradient
    CLOCK.tick("cggm - gradientDescent - gradient");
    Eigen::VectorXd grad = gradient(R, A, p, u, R_star_0_inv, S, UWU, lambda_cpath, k, -1);
    CLOCK.tock("cggm - gradientDescent - gradient");

    // Compute step size interval that keeps the solution in the domain
    CLOCK.tick("cggm - gradientDescent - maxStepSize");
    Eigen::VectorXd step_sizes = maxStepSize(R, A, p, R_star_0_inv, grad, k);
    CLOCK.tock("cggm - gradientDescent - maxStepSize");

    // Compute optimal step size, let the minimum step size be 0 instead of
    // negative
    CLOCK.tick("cggm - gradientDescent - gssStepSize");
    double step_size = gssStepSize(R, A, p, u, R_star_0_inv, S, UWU, grad, lambda_cpath, k, 0, step_sizes(1), gss_tol);
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


/*Eigen::MatrixXd updateInverse(const Eigen::MatrixXd& inverse,
                              const Eigen::VectorXd& vec, int m)
{
    // Sherman-Morrison with A = inverse, u = e_m, v = vec
    Eigen::VectorXd Au = inverse.col(m);
    Eigen::VectorXd vA = vec.transpose() * inverse;
    Eigen::MatrixXd N = Au * vA.transpose();
    double D = 1.0 / (1.0 + vA(m));
    Eigen::MatrixXd result = inverse - D * N;

    // Sherman-Morrison with A = result, u = vec, v = e_m
    Au.noalias() = result * vec;
    vA = result.row(m);
    N.noalias() = Au * vA.transpose();
    D = 1.0 / (1.0 + vA.dot(vec));
    result -= D * N;

    return result;
}*/


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


int fusionChecks(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
                 const Eigen::VectorXi& p, const Eigen::VectorXi& u,
                 const Eigen::MatrixXd& R_star_0_inv,
                 const Eigen::MatrixXd& S, const Eigen::MatrixXd& UWU,
                 double lambda_cpath, int k, double fusion_check_threshold,
                 int verbose)
{
    // If lambda is zero no need to check for fusions
    if (lambda_cpath <= 0) {
        return -1;
    }

    // Number of clusters
    int n_clusters = R.cols();
    int n_variables = S.cols();

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
        if (normRA(R, A, p, k, m) > std::sqrt(n_variables) * fusion_check_threshold) continue;

        // Set row/column k to the values in m
        // setEqualToClusterInplace(R_new, A_new, p, k, m);

        // Test with setting k and m to the average of these rows/cols
        CLOCK.tick("cggm - fusionChecks - setEqualToClusterMeansInplace");
        setEqualToClusterMeansInplace(R_new, A_new, p, k, m);
        CLOCK.tock("cggm - fusionChecks - setEqualToClusterMeansInplace");
        /*CLOCK.tick("cggm - fusionChecks - computeRStar0Inv");
        Eigen::MatrixXd R_star_0_inv_new = computeRStar0Inv(R_new, A_new, p, k);
        CLOCK.tock("cggm - fusionChecks - computeRStar0Inv");*/
        CLOCK.tick("cggm - fusionChecks - updateRStar0Inv");
        Eigen::MatrixXd R_star_0_inv_new = updateRStar0Inv(R_star_0_inv, R, R_new, A, A_new, p, k, m);
        CLOCK.tock("cggm - fusionChecks - updateRStar0Inv");

        //Rcpp::Rcout << "\nR_star_0_inv_new:\n";
        //printMatrix(R_star_0_inv_new);
        //printMatrix(updateRStar0Inv(R_star_0_inv, R, R_new, A, A_new, p, k, m));

        // Compute gradient, this time there is a fusion candidate
        CLOCK.tick("cggm - fusionChecks - gradient");
        Eigen::VectorXd grad = gradient(R_new, A_new, p, u, R_star_0_inv_new, S, UWU, lambda_cpath, k, m);
        CLOCK.tock("cggm - fusionChecks - gradient");

        // Modify the (m+1)th value of G
        G(1 + m) -= 1;

        // Take the "inverse" of G
        for (int i = 0; i < G.size(); i++) {
            if (G(i) == 0) continue;
            G(i) = 1.0 / G(i);
        }

        // Compute lhs and rhs of the test
        double test_lhs = std::sqrt((grad.array() * grad.array() * G).sum());
        double test_rhs = lambda_cpath * UWU(k, m);

        if (verbose > 0) {
            Rcpp::Rcout << "test fusion of row/columns " << k + 1 << " and " << m + 1 << ":\n";
            Rcpp::Rcout << "    " << test_lhs << " <= " << test_rhs;
            if (test_lhs <= test_rhs) Rcpp::Rcout << ": TRUE\n";
            if (test_lhs > test_rhs) Rcpp::Rcout << ": FALSE\n";
        }

        // Perform the test
        if (test_lhs <= test_rhs) {
            return m;
        }

        // Undo changes for the next iteration
        if (!((m == n_clusters - 1) && ((m == n_clusters - 2) && (k == n_clusters - 1)))) {
            A_new(k) = A(k);
            R_new.row(k) = R.row(k);
            R_new.col(k) = R.col(k);
            for (int i = 0; i < G.size(); i++) {
                if (G(i) == 0) continue;
                G(i) = 1.0 / G(i);
            }
            G(1 + m) += 1;
        }
    }

    // Result stores the index that k should fuse with, if no fusion should be
    // done, the result is -1
    return -1;
}


void performFusion(Eigen::MatrixXd& R, Eigen::VectorXd& A, Eigen::VectorXi& p,
                   Eigen::VectorXi& u, Eigen::MatrixXd& UWU, int k, int target,
                   int verbose)
{
    if (verbose > 0) {
        Rcpp::Rcout << "fuse row/column " << k + 1<< " with " << target + 1 << "\n\n";
    }

    // Preliminaries
    int n_variables = u.size();
    int n_clusters = p.size();

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

    /*Rcpp::Rcout << "\nu: [";
     for (int i = 0; i < n_variables - 1; i++) {
     Rcpp::Rcout << u(i) << " ";
     }
     Rcpp::Rcout << u(n_variables - 1) << "]\n";*/

    // Fusion of rows/cols in UWU
    UWU(target, target) += UWU(k, k) + UWU(k, target) + UWU(target, k);

    for (int i = 0; i < n_clusters; i++) {
        if (i == k || i == target) continue;

        UWU(i, target) += UWU(i, k);
        UWU(target, i) += UWU(i, k);
    }

    // Drop row/column k
    dropVariableInplace(UWU, k);

    /*Rcpp::Rcout << "UWU: [";
     for (int i = 0; i < UWU.rows(); i++) {
     if (i > 0) Rcpp::Rcout << "      ";

     for (int j = 0; j < UWU.cols() - 1; j++) {
     Rcpp::Rcout << UWU(i, j) << ' ';
     }
     if (i < UWU.cols() - 1) Rcpp::Rcout << UWU(i, UWU.cols() - 1) << '\n';
     if (i == UWU.cols() - 1) Rcpp::Rcout << UWU(i, UWU.cols() - 1) << "]\n";
     }*/

    Rcpp::Rcout << "R: [";
    for (int i = 0; i < R.rows(); i++) {
        if (i > 0) Rcpp::Rcout << "    ";

        for (int j = 0; j < R.cols() - 1; j++) {
            Rcpp::Rcout << R(i, j) << ' ';
        }
        if (i < R.cols() - 1) Rcpp::Rcout << R(i, R.cols() - 1) << '\n';
        if (i == R.cols() - 1) Rcpp::Rcout << R(i, R.cols() - 1) << "]\n";
    }

    // Fusions in R and A
    if (p(target) <= 1) {
        R(target, target) = R(k, target);
    }
    dropVariableInplace(R, k);
    dropVariableInplace(A, k);

    Rcpp::Rcout << "R: [";
    for (int i = 0; i < R.rows(); i++) {
        if (i > 0) Rcpp::Rcout << "    ";

        for (int j = 0; j < R.cols() - 1; j++) {
            Rcpp::Rcout << R(i, j) << ' ';
        }
        if (i < R.cols() - 1) Rcpp::Rcout << R(i, R.cols() - 1) << '\n';
        if (i == R.cols() - 1) Rcpp::Rcout << R(i, R.cols() - 1) << "]\n";
    }

    Rcpp::Rcout << "A: [";
    for (int i = 0; i < A.size() - 1; i++) {
        Rcpp::Rcout << A(i) << ' ';
    }
    Rcpp::Rcout << A(A.size() - 1) << "]\n";

    // Increase cluster size of the target cluster
    p(target) += p(k);

    // Move cluster sizes of clusters with index larger than k one position to
    // the left
    for (int i = k; i < n_clusters - 1; i++) {
        p(i) = p(i + 1);
    }
    p.conservativeResize(n_clusters - 1);

    /*Rcpp::Rcout << "p: [";
     for (int i = 0; i < p.size() - 1; i++) {
     Rcpp::Rcout << p(i) << ' ';
     }
     Rcpp::Rcout << p(p.size() - 1) << "]\n";*/
}


// [[Rcpp::export]]
Rcpp::List cggm(const Eigen::MatrixXd& Ri, const Eigen::VectorXd& Ai,
                const Eigen::VectorXi& pi, const Eigen::VectorXi& ui,
                const Eigen::MatrixXd& S, const Eigen::MatrixXd& UWUi,
                const Eigen::VectorXd& lambdas, double gss_tol, double conv_tol,
                double fusion_check_threshold, int max_iter, bool store_all_res,
                int verbose)
{
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
            for (int k = 0; k < R.cols(); k++) {
                Eigen::MatrixXd R_star_0_inv = computeRStar0Inv(R, A, p, k);

                // Check if there is an eligible fusion
                int fusion_index = fusionChecks(
                    R, A, p, u, R_star_0_inv, S, UWU, lambdas(lambda_index), k,
                    fusion_check_threshold, verbose
                );

                // No eligible fusions, proceed to gradient descent
                if (fusion_index < 0) {
                    gradientDescent(R, A, p, u, R_star_0_inv, S, UWU, lambdas(lambda_index), k, gss_tol, verbose);
                } else {
                    performFusion(R, A, p, u, UWU, k, fusion_index, verbose);
                    break;
                }
            }

            l0 = l1;
            l1 = lossRA(R, A, p, u, S, UWU, lambdas(lambda_index));
            iter++;

            if (verbose > 0) {
                Rcpp::Rcout << "lambda: " << lambdas(lambda_index) << " | iteration: ";
                Rcpp::Rcout << iter << " | loss: " << l1 << "\n\n";
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

    return results.convertToRcppList();
    return Rcpp::List::create(Rcpp::Named("R") = R,
                              Rcpp::Named("A") = A,
                              Rcpp::Named("p") = p,
                              Rcpp::Named("u") = u,
                              Rcpp::Named("UWU") = UWU);
}
