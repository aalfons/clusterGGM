#include <RcppEigen.h>
#include "utils2.h"
#include "variables.h"


double square2(double x)
{
    return x * x;
}


Eigen::SparseMatrix<double>
convertToSparse(const Eigen::MatrixXd& W_keys, const Eigen::VectorXd& W_values,
                int n_variables)
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


//Eigen::MatrixXd
void
computeRStar0Inv2(const Variables& vars, int k)
{
    // Create references to the relevant variables in the struct
    const Eigen::MatrixXd &R = vars.m_R;
    const Eigen::MatrixXd &A = vars.m_A;
    const Eigen::VectorXi &p = vars.m_p;

    // Number of rows/columns of R
    int n_clusters = R.cols();
/*
    // Drop row/column k from R
    Eigen::MatrixXd result = dropVariable(R, k);

    for (int i = 0; i < n_clusters; i++) {
        if (i == k) {
            continue;
        }

        result(i - (i > k), i - (i > k)) += (A(i) - R(i, i)) / p(i);
    }

    result = result.inverse();

    return result;*/
}
