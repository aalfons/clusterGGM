#include <RcppEigen.h>
#include "utils.h"
#include "variables.h"


// [[Rcpp::export(.compute_Theta)]]
Eigen::MatrixXd compute_Theta(const Eigen::MatrixXd& R,
                              const Eigen::VectorXd& A,
                              const Eigen::VectorXi& u)
{
    // Preliminaries
    int n_variables = u.size();
    int n_clusters = R.cols();
    Eigen::MatrixXd result(n_variables, n_variables);

    // Fill in R
    for (int j = 0; j < n_variables; j++){
        for (int i = 0; i < n_variables; i++) {
            result(i, j) = R(u(i), u(j));
        }
    }

    // Add diagonal component
    for (int i = 0; i < n_variables; i++) {
        result(i, i) = A(u(i));
    }

    return result;
}


double square(double x)
{
    return x * x;
}


Eigen::MatrixXd drop_variable(const Eigen::MatrixXd& X, int k)
{
    /* Drop row and column from square matrix
     *
     * Inputs:
     * X: matrix
     * k: index of row/column to be removed
     *
     * Output
     * Matrix with one fewer row and column
     */

    // Number of rows/columns of X
    int n = X.rows();

    // Initialize result
    Eigen::MatrixXd result(n - 1, n - 1);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (j == k || i == k) {
                continue;
            }

            result(i - (i > k), j - (j > k)) = X(i, j);
        }
    }

    return result;
}


void drop_variable_inplace(Eigen::MatrixXd& X, int k)
{
    /* Drop row and column from a square matrix in place
     *
     * Inputs:
     * X: matrix
     * k: index of row/column to be removed
     */

    // Number of rows/columns
    int n = X.rows();

    // Shift each with index larger than k one position upwards
    for (int j = 0; j < n; j++) {
        for (int i = k; i < n - 1; i++) {
            X(i, j) = X(i + 1, j);
        }
    }

    // Transpose X and do it again
    X.transposeInPlace();

    // Shift each with index larger than k one position upwards
    for (int j = 0; j < n; j++) {
        for (int i = k; i < n - 1; i++) {
            X(i, j) = X(i + 1, j);
        }
    }

    // Resize
    X.conservativeResize(n - 1, n - 1);
}


Eigen::SparseMatrix<double>
convert_to_sparse(const Eigen::MatrixXd& W_keys,
                  const Eigen::VectorXd& W_values, int n_variables)
{
    /* Convert key value pairs into a sparse weight matrix.
     *
     * Inputs:
     * W_keys: indices for the nonzero elements of the weight matrix
     * W_values: nonzero elements of the weight matrix
     * n_variables: number of variables used to construct the weight matrix
     *
     * Output:
     * Sparse weight matrix
     */

    // Number of nnz elements
    int nnz = W_keys.cols();

    // Initialize list of triplets
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);

    // Fill list of triplets
    for(int i = 0; i < nnz; i++) {
        // If the row and column indices are the same, ignore the value as it
        // is not of importance
        if (W_keys(0, i) == W_keys(1, i)) {
            continue;
        }

        // Add weight and store both upper and lower triangular parts
        triplets.push_back(
            Eigen::Triplet<double>(W_keys(0, i), W_keys(1, i), W_values(i))
        );
    }

    // Construct the sparse matrix
    Eigen::SparseMatrix<double> result(n_variables, n_variables);
    result.setFromTriplets(triplets.begin(), triplets.end());

    return result;
}


Eigen::MatrixXd compute_R_star0_inv(const Variables& vars, int k)
{
    /* Compute the inverse of R* excluding the kth row and column
     *
     * Inputs:
     * vars: struct containing the optimization variables
     * k: cluster of interest
     *
     * Output:
     * The inverse of R* minus row/column k
     */

    // Get R* from the variables
    Eigen::MatrixXd result = drop_variable(vars.m_Rstar, k);

    // Compute inverse
    result = result.inverse();

    return result;
}


void drop_variable_inplace(Eigen::VectorXd& x, int k)
{
    /* Drop element from vector in place
     *
     * Inputs:
     * x: vector
     * k: index of element to be removed
     */
    // Number of rows/columns of X
    int n = x.size();

    // Shift elements
    for (int i = k; i < n - 1; i++) {
        x(i) = x(i + 1);
    }

    // Resize
    x.conservativeResize(n - 1);
}


Eigen::VectorXd drop_variable(const Eigen::VectorXd& x, int k)
{
    /* Drop the kth element from a vector
     *
     * Inputs:
     * x: vector
     * k: index of element to be dropped
     *
     * Output:
     * Vector that has 1 fewer element
     */

    // Number of elements in x
    int n = x.size();

    // Initialize result
    Eigen::VectorXd result(n - 1);

    for (int i = 0; i < n; i++) {
        if (i == k) {
            continue;
        }

        result(i - (i > k)) = x(i);
    }

    return result;
}


double partial_trace(const Eigen::MatrixXd& S, const Eigen::VectorXi& u, int k)
{
    /* Compute the trace of S only for variables that belong to cluster k
     *
     * Inputs:
     * S: sample covariance matrix
     * k: cluster of interest
     *
     * Output:
     * Partial trace
     */

    // Number of elements on the diagonal
    int P = S.cols();

    // Initialize result
    double result = 0;

    for (int i = 0; i < P; i++) {
        if (u(i) != k) continue;

        result += S(i, i);
    }

    return result;
}


double sum_selected_elements(const Eigen::MatrixXd& S, const Eigen::VectorXi& u,
                             const Eigen::VectorXi& p, int k)
{
    /* Compute U[, k] * S * U[, k]
     *
     * Inputs:
     * S: sample covariance matrix
     * u: membership vector
     * p: vector of cluster sizes
     * k: cluster of interest
     *
     * Output:
     * The sum of the selected elements of S
     */

    // Number of rows/columns in S
    int K = S.cols();

    // Number of nonzero indices
    int nnz = p(k);

    // Vector that holds the indices of where u = k
    Eigen::VectorXi indices(nnz);
    int idx = 0;

    for (int i = 0; i < K; i++) {
        if (u(i) != k) continue;

        // Store index and increment the index of the indices vector by one
        indices(idx) = i;
        idx++;

        if (idx == nnz) break;
    }

    // Initialize result
    double result = 0;

    for (int i = 0; i < nnz; i++) {
        for (int j = 0; j < nnz; j++) {
            result += S(indices(j), indices(i));
        }
    }

    return result;
}


Eigen::VectorXd
sum_multiple_selected_elements(const Eigen::MatrixXd& S,
                               const Eigen::VectorXi& u,
                               const Eigen::VectorXi& p, int k)
{
    /* Compute U[, k] * S * U[, -k]
     *
     * Inputs:
     * S: sample covariance matrix
     * u: membership vector
     * p: vector of cluster sizes
     * k: cluster of interest
     *
     * Output:
     * Vector of the sums of the selected elements of S
     */

    // Preliminaries
    int n_clusters = p.size();
    int n_variables = u.size();

    // Vector that holds sum of all elements before the ith element
    Eigen::VectorXi start_index(n_clusters);
    start_index(0) = 0;
    for (int i = 1; i < n_clusters; i++) {
        start_index(i) = start_index(i - 1) + p(i - 1);
    }

    // Vector with the indices of the variables sorted by cluster
    Eigen::VectorXi indices(n_variables);

    // Fill the vector
    Eigen::VectorXi current_index = Eigen::VectorXi::Zero(n_clusters);
    for (int i = 0; i < n_variables; i++) {
        indices(start_index(u(i)) + current_index(u(i))) = i;
        current_index(u(i))++;
    }

    // Compute U[, k] * S, ignore elements of the result that are not used
    // elsewhere
    std::set<int> ignore_idx;           // This can be done more efficiently with a binary
    for (int i = 0; i < p(k); i++) {    // search, as the indices per cluster are sorted
        ignore_idx.insert(indices(start_index(k) + i));
    }

    Eigen::VectorXd intermediate_result = Eigen::VectorXd::Zero(n_variables);
    for (int i = 0; i < n_variables; i++) {
        if (ignore_idx.find(i) != ignore_idx.end()) continue;

        for (int j = 0; j < p(k); j++) {
            intermediate_result(i) += S(indices(start_index(k) + j), i);
        }
    }

    // Now compute (U[, k] * S) * U[, -k]
    Eigen::VectorXd result = Eigen::VectorXd::Zero(n_clusters - 1);

    for (int i = 0; i < n_clusters; i++) {
        if (i == k) continue;

        for (int j = 0; j < p(i); j++) {
            result(i - (i > k)) +=
                intermediate_result(indices(start_index(i) + j));
        }
    }

    return result;
}


std::pair<Eigen::MatrixXd, Eigen::VectorXd>
update_RA(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
          const Eigen::VectorXd& values, int k)
{
    // Initialize result
    Eigen::MatrixXd R_new(R);
    Eigen::VectorXd A_new(A);

    // The updating
    update_RA_inplace(R_new, A_new, values, k);

    return std::make_pair(R_new, A_new);
}


void update_RA_inplace(Eigen::MatrixXd& R, Eigen::VectorXd& A,
                       const Eigen::VectorXd& values, int k)
{
    // Number of clusters
    int n_clusters = R.cols();

    // The updating
    A(k) += values(0);
    R.col(k) += values.tail(n_clusters);
    R.row(k) += values.tail(n_clusters);
    R(k, k) -= values(1 + k);
}
