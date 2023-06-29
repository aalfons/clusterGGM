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


void printVector(const Eigen::VectorXd& vec)
{
    Rcpp::Rcout << '[';
    for (int i = 0; i < vec.size() - 1; i++) {
        Rcpp::Rcout << vec(i) << ' ';
    }
    Rcpp::Rcout << vec(vec.size() - 1) << "]\n";
}


void printVector(const Eigen::ArrayXd& vec)
{
    Rcpp::Rcout << '[';
    for (int i = 0; i < vec.size() - 1; i++) {
        Rcpp::Rcout << vec(i) << ' ';
    }
    Rcpp::Rcout << vec(vec.size() - 1) << "]\n";
}


void printMatrix(const Eigen::MatrixXd& mat)
{
    Rcpp::Rcout << "[";
    for (int i = 0; i < mat.rows(); i++) {
        if (i > 0) Rcpp::Rcout << ' ';
        
        for (int j = 0; j < mat.cols() - 1; j++) {
            Rcpp::Rcout << mat(i, j) << ' ';
        }
        if (i < mat.cols() - 1) Rcpp::Rcout << mat(i, mat.cols() - 1) << '\n';
        if (i == mat.cols() - 1) Rcpp::Rcout << mat(i, mat.cols() - 1) << "]\n";
    }
}


// [[Rcpp::export]]
Eigen::MatrixXd updateInverse(const Eigen::MatrixXd& inverse,
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
        double test_lhs =_delegating_constructors
# define __cpp_delegating_constructors 200604L
#endif

#ifndef __FLT32_HAS_INFINITY__
# define __FLT32_HAS_INFINITY__ 1
#endif

#ifndef __DBL_MAX__
# define __DBL_MAX__ double(1.79769313486231570814527423731704357e+308L)
#endif

#ifnde