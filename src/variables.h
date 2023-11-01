#ifndef VARIABLES_H
#define VARIABLES_H

#include <RcppEigen.h>
#include "utils2.h"


struct Variables {
    Eigen::SparseMatrix<double> m_D;
    Eigen::MatrixXd m_R;
    Eigen::VectorXd m_A;

    Variables(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
              const Eigen::SparseMatrix<double>& W, const Eigen::VectorXi& p)
    {
        // Set attributes
        m_R = R;
        m_A = A;

        // Compute the distance matrix for the first time
        setDistances(W, p);
    }

    double distance(int i, int j, const Eigen::VectorXi& p)
    {
        /* Compute the distance between two clusters
         *
         * Inputs:
         * i: index of one cluster
         * j: index of another cluster
         * p: vector of cluster sizes
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

            result += p(k) * square2(m_R(k, i) - m_R(k, j));
        }

        result += (p(i) - 1) * square2(m_R(i, i) - m_R(j, i));
        result += (p(j) - 1) * square2(m_R(j, j) - m_R(j, i));

        return result;
    }

    void updateAllDistances(const Eigen::VectorXi& p)
    {
        /* Update the values in the existing distance matrix
         *
         * Input:
         * p: vector of cluster sizes
         */

        // Update the values for the distances
        for (int j = 0; j < m_D.outerSize(); j++) {
            Eigen::SparseMatrix<double>::InnerIterator it(m_D, j);

            for (; it; ++it) {
                // Row index
                int i = it.row();

                // Compute distance
                it.valueRef() = distance(i, j, p);
            }
        }
    }

    void setDistances(const Eigen::SparseMatrix<double>& W,
                      const Eigen::VectorXi& p)
    {
        /* Construct and fill a sparse distance matrix.
         *
         * Inputs:
         * W: sparse weight matrix
         * p: vector of cluster sizes
         */

        // Copy W to get the same sparsity structure
        m_D = W;

        // Set the distances between the clusters for which there is a nonzero
        // weight
        updateAllDistances(p);
    }
};

#endif // VARIABLES_H
