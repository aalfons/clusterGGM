#include <RcppArmadillo.h>
#include <map>
#include "utils.h"
#include "variables.h"


// [[Rcpp::export(.compute_Theta)]]
arma::mat compute_Theta(const arma::mat& R,
                        const arma::vec& A,
                        const arma::ivec& u)
{
    // Preliminaries
    int n_variables = u.n_elem;
    arma::mat result(n_variables, n_variables);

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


arma::mat drop_variable(const arma::mat& X, int k)
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
    int n = X.n_rows;

    // Initialize result
    arma::mat result(n - 1, n - 1);

    if (n == 1) {
      return result;
    }

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


void drop_variable_inplace(arma::mat& X, int k)
{
    /* Drop row and column from a square matrix in place
     *
     * Inputs:
     * X: matrix
     * k: index of row/column to be removed
     */
    X.shed_row(k);
    X.shed_col(k);
}


arma::sp_mat
convert_to_sparse(const arma::mat& W_keys,
                  const arma::vec& W_values, int n_variables)
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
    int nnz = W_keys.n_cols;

    std::map<std::pair<arma::uword, arma::uword>, double> entries;

    for(int i = 0; i < nnz; i++) {
        // If the row and column indices are the same, ignore the value as it
        // is not of importance
        if (W_keys(0, i) == W_keys(1, i)) {
            continue;
        }

        // Add weight and store both upper and lower triangular parts
        arma::uword row = (arma::uword) W_keys(0, i);
        arma::uword col = (arma::uword) W_keys(1, i);
        entries[{row, col}] += W_values(i);
    }

    // Construct the sparse matrix
    arma::umat locations(2, entries.size());
    arma::vec values(entries.size());
    arma::uword idx = 0;
    for (const auto& entry : entries) {
        locations(0, idx) = entry.first.first;
        locations(1, idx) = entry.first.second;
        values(idx) = entry.second;
        idx++;
    }

    arma::sp_mat result(locations, values, n_variables, n_variables);

    return result;
}


arma::mat compute_R_star0_inv(const Variables& vars, int k)
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
    arma::mat result = drop_variable(vars.m_Rstar, k);

    // Compute inverse
    result = arma::inv_sympd(result);

    return result;
}


void drop_variable_inplace(arma::vec& x, int k)
{
    /* Drop element from vector in place
     *
     * Inputs:
     * x: vector
     * k: index of element to be removed
     */
    x.shed_row(k);
}


arma::vec drop_variable(const arma::vec& x, int k)
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
    int n = x.n_elem;

    // Initialize result
    arma::vec result(n - 1);

    for (int i = 0; i < n; i++) {
        if (i == k) {
            continue;
        }

        result(i - (i > k)) = x(i);
    }

    return result;
}


void update_inverse_inplace(arma::mat& M_inv, const arma::mat& M,
                            int k)
{
    /* Given a symmetric K by K matrix M and the inverse of M excluding
     * row/column k+1, compute the inverse of M excluding row/column k. If k is
     * the last row/column, it wraps around and k+1 becomes 0.
     *
     * Inputs:
     * M_inv: the inverse of M[-(k+1), -(k+1)] (M without row/column k+1).
     * M: the original K by K matrix.
     * k: the row/column of k that should be removed. The row/column that is
     * added back is k+1.
     *
     * At the end, M_inv is the inverse of M[-k, -k]
     */
    // The index arithmetic below assumes at least one row/column is present
    if (M_inv.n_cols < 1) {
        return;
    }

    // Row/column being removed
    int k0 = k - 1;

    // Row/column being added
    int k1 = k;

    // Check if k is the last column. If so, move the first column to the last
    // position
    if (k0 < 0) {
        // Cyclically shift rows and columns so column/row 0 moves to the end.
        if (M_inv.n_cols > 1) {
            M_inv = arma::shift(arma::shift(M_inv, -1, 0), -1, 1);
        }

        // Set k0
        k0 = (int) M_inv.n_cols;
    }

    // The difference between M[-k0, -k0] and M[-k1, -k1]
    arma::vec update = M.col(k0) - M.col(k1);

    // Halve the value on the diagonal, because if we add the difference to the
    // row and column, the diagonal difference is added twice
    update(k0) = (M(k0, k0) - M(k1, k1)) / 2.0;

    // Remove the value at the k0th index of the difference, as this element
    // disappears
    drop_variable_inplace(update, k1);

    // Procedure to change the inverse after changing the column
    arma::vec Au = M_inv.col(k0 - (k0 == (int) M_inv.n_cols));
    arma::vec vA = (update.t() * M_inv).t();
    arma::mat N = Au * vA.t();
    double D = 1.0 / (1.0 + vA(k0 - (k0 == (int) M_inv.n_cols)));
    M_inv -= D * N;

    // Procedure to change the inverse after changing the row
    Au = M_inv * update;
    vA = M_inv.row(k0 - (k0 == (int) M_inv.n_cols)).t();
    N = Au * vA.t();
    D = 1.0 / (1.0 + Au(k0 - (k0 == (int) M_inv.n_cols)));
    M_inv -= D * N;

    // Check quality of inverse, first select a column of M
    arma::vec m_0 = M.col(0 + (k == 0));
    drop_variable_inplace(m_0, k);

    // Dot product of column 0 of M and its inverse should be very close to 1
    if (std::fabs(arma::dot(m_0, M_inv.col(0)) - 1.0) > 1e-7) {
        // Drop variable k
        M_inv = drop_variable(M, k);

        // Compute inverse
        M_inv = arma::inv_sympd(M_inv);
    }
}


// [[Rcpp::export()]]
arma::mat update_inverse(const arma::mat& M_inv,
                         const arma::mat& M, int k)
{
    arma::mat result(M_inv);
    update_inverse_inplace(result, M, k);
    return result;
}


double partial_trace(const arma::mat& S, const arma::ivec& u, int k)
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
    int P = S.n_cols;

    // Initialize result
    double result = 0;

    for (int i = 0; i < P; i++) {
        if (u(i) != k) continue;

        result += S(i, i);
    }

    return result;
}


double sum_selected_elements(const arma::mat& S, const arma::ivec& u,
                             const arma::ivec& p, int k)
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

    arma::uvec idx = arma::find(u == k);

    return arma::accu(S.submat(idx, idx));
}


arma::vec
sum_multiple_selected_elements(const arma::mat& S,
                               const arma::ivec& u,
                               const arma::ivec& p, int k)
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

    // Column sums of S restricted to the rows belonging to cluster k
    arma::uvec idx_k = arma::find(u == k);
    arma::rowvec col_sums = arma::sum(S.rows(idx_k), 0);

    // Accumulate those sums per cluster i != k
    arma::vec result(p.n_elem - 1, arma::fill::zeros);

    for (int i = 0; i < (int) u.n_elem; i++) {
        if (u(i) == k) continue;

        result(u(i) - (u(i) > k)) += col_sums(i);
    }

    return result;
}


void update_RA_inplace(arma::mat& R, arma::vec& A,
                       const arma::vec& values, int k)
{
    // Number of clusters
    int n_clusters = R.n_cols;

    // The updating
    A(k) += values(0);
    R.col(k) += values.tail(n_clusters);
    R.row(k) += values.tail(n_clusters).t();
    R(k, k) -= values(1 + k);
}
