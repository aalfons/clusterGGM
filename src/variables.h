#ifndef VARIABLES_H
#define VARIABLES_H

#include <RcppEigen.h>
#include "utils2.h"


struct Variables {
    Eigen::SparseMatrix<double> m_D;
    Eigen::MatrixXd m_R;
    Eigen::MatrixXd m_Rstar;
    Eigen::VectorXd m_A;
    Eigen::VectorXi m_p;
    Eigen::VectorXi m_u;

    Variables(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
              const Eigen::SparseMatrix<double>& W, const Eigen::VectorXi& p,
              const Eigen::VectorXi& u)
    {
        // Set attributes
        m_R = R;
        m_A = A;
        m_p = p;
        m_u = u;

        // Compute R*
        m_Rstar = R;
        for (int i = 0; i < R.cols(); i++) {
            m_Rstar(i, i) += (A(i) - R(i, i)) / p(i);
        }

        // Compute the distance matrix for the first time
        setDistances(W);
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

        return std::sqrt(result);
    }

    void updateAllDistances()
    {
        /* Update the values in the existing distance matrix */

        // Update the values for the distances
        for (int j = 0; j < m_D.outerSize(); j++) {
            Eigen::SparseMatrix<double>::InnerIterator it(m_D, j);

            for (; it; ++it) {
                // Row index
                int i = it.row();

                // Compute distance
                it.valueRef() = distance(i, j);
            }
        }
    }

    void setDistances(const Eigen::SparseMatrix<double>& W)
    {
        /* Construct and fill a sparse distance matrix.
         *
         * Inputs:
         * W: sparse weight matrix
         */

        // Copy W to get the same sparsity structure
        m_D = W;

        // Set the distances between the clusters for which there is a nonzero
        // weight
        updateAllDistances();
    }

    void updateCluster(const Eigen::VectorXd& values,
                       const Eigen::SparseMatrix<double>& E, int k)
    {
        /* Update elements of R and A that correspond to cluster k. Also update
         * the distances and R*
         *
         * Inputs:
         * values: update in the form [a_kk, r_k]
         * k: cluster of interest
         */

        // Update the values of R and A
        updateRAInplace2(m_R, m_A, values, k);

        // Update the distances
        for (int j = 0; j < m_D.outerSize(); j++) {
            // Iterators
            Eigen::SparseMatrix<double>::InnerIterator D_it(m_D, j);
            Eigen::SparseMatrix<double>::InnerIterator E_it(E, j);

            for (; D_it; ++D_it) {
                // Index
                int i = D_it.row();

                // If i and j are not equal to k, there is a more efficient
                // approach to updating the weights
                if (i == k || j == k) {
                    D_it.valueRef() = distance(i, j);
                } else {
                    // Compute distance
                    double d_ij = E_it.value();
                    d_ij += m_p(k) * square2(m_R(i, k) - m_R(j, k));
                    D_it.valueRef() = std::sqrt(d_ij);
                }

                // Continue iterator for E
                ++E_it;
            }
        }

        // Update R*
        m_Rstar.row(k) = m_R.row(k);
        m_Rstar.col(k) = m_R.col(k);
        m_Rstar(k, k) += (m_A(k) - m_R(k, k)) / m_p(k);

        // Compute R*
        /*m_Rstar = m_R;
        for (int i = 0; i < m_R.cols(); i++) {
            m_Rstar(i, i) += (m_A(i) - m_R(i, i)) / m_p(i);
        }*/
    }
};

#endif // VARIABLES_H
