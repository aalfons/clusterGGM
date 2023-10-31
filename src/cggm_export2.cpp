#include <RcppEigen.h>
#include <iostream>
#include "utils2.h"


class Variables {
    Eigen::SparseMatrix<double> m_W;
    Eigen::SparseMatrix<double> m_D;
    Eigen::MatrixXd m_R;
    Eigen::VectorXd m_A;
    Eigen::VectorXi m_p;

    Variables(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
              const Eigen::SparseMatrix<double>& W, const Eigen::VectorXi& p)
    {
        // Set attributes
        m_R = R;
        m_A = A;
        m_W = W;
        m_p = p;

        // Compute the distance matrix for the first time
        setDistances();
    }

    double distance(int i, int j)
    {
        /* Compute the distance between two clusters
         *
         * Inputs:
         * i: index of one cluster
         * j: index of another cluster
         *
         * Output:
         * The distance
         */

        // Number of rows/cols of R
        int n_clusters = m_R.rows();

        // Initialize result
        double result = square2(m_A(i) - m_A(j));

        for (int k = 0; k < n_clusters; k++) {
            if (k == i || k == j) {
                continue;
            }

            result += m_p(k) * square2(m_R(k, i) - m_R(k, j));
        }

        result += (m_p(i) - 1) * square2(m_R(i, i) - m_R(j, i));
        result += (m_p(j) - 1) * square2(m_R(j, j) - m_R(j, i));

        return result;
    }

    void updateDistances()
    {
        /* Update the values in the existing distance matrix */

        // Update the values for the distances
        for (int j = 0; j < m_W.outerSize(); j++) {
            Eigen::SparseMatrix<double>::InnerIterator it(m_W, j);

            for (; it; ++it) {
                // Row index
                int i = it.row();

                // Compute distance
                it.valueRef() = distance(i, j);
            }
        }
    }

    void setDistances()
    {
        /* Construct and fill a sparse distance matrix. */

        // Copy W to get the same sparsity structure
        m_D = m_W;

        // Set the distances between the clusters for which there is a nonzero
        // weight
        updateDistances();
    }

    double loss()
    {

    }
};


Eigen::SparseMatrix<double>
convertToSparse(const Eigen::MatrixXd& W_keys, const Eigen::VectorXd& W_values,
                int n_variables)
{
    /* Convert key value pairs into a sparse weight matrix.
     *
     * Inputs:
     * W_keys: indices for the nonzero elements of the weight matrix (2 x nnz)
     * W_values: nonzero elements of the weight matrix (nnz)
     * n_variables: number of variables used to construct the weight matrix
     *
     * Output:
     * sparse weight matrix
     */

    // Number of nnz elements
    int nnz = W_keys.cols();

    // Initialize list of triplets
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);

    // Fill list of triplets
    for(int i = 0; i < nnz; i++) {
        triplets.push_back(
            Eigen::Triplet<double>(W_keys(0, i), W_keys(1, i), W_values(i))
        );
    }

    // Construct the sparse matrix
    Eigen::SparseMatrix<double> result(n_variables, n_variables);
    result.setFromTriplets(triplets.begin(), triplets.end());

    return result;
}


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


// [[Rcpp::export]]
void test(const Eigen::MatrixXd& W_keys, const Eigen::VectorXd& W_values, int p,
          const Eigen::VectorXi& u)
{
    /* Inputs:
     * W_keys: indices for the nonzero elements of the weight matrix (2 x p)
     * W_values: nonzero elements of the weight matrix (p)
     *
     */

    auto W = convertToSparse(W_keys, W_values, p);
    std::cout << W << '\n';

    W = fuseW(W, u);
    std::cout << W << '\n';
}







