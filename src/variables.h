#ifndef VARIABLES_H
#define VARIABLES_H

#include <RcppEigen.h>
#include "utils.h"


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
        set_distances(W);
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
        double result = square(m_A(i) - m_A(j));

        for (int k = 0; k < n_clusters; k++) {
            if (k == i || k == j) {
                continue;
            }

            result += m_p(k) * square(m_R(k, i) - m_R(k, j));
        }

        result += (m_p(i) - 1) * square(m_R(i, i) - m_R(j, i));
        result += (m_p(j) - 1) * square(m_R(j, j) - m_R(j, i));

        return std::sqrt(result);
    }

    void update_all_distances()
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

    void set_distances(const Eigen::SparseMatrix<double>& W)
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
        update_all_distances();
    }

    void update_cluster(const Eigen::VectorXd& values,
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
        update_RA_inplace(m_R, m_A, values, k);

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
                    d_ij += m_p(k) * square(m_R(i, k) - m_R(j, k));
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
    }

    void fuse_clusters(int k, int m, const Eigen::SparseMatrix<double>& W)
    {
        /* Fuse clusters k and m, m is the index that is dropped from the
         * variables
         *
         * Inputs:
         * k: cluster of interest
         * m: cluster k is fused with
         * W: sparse weight matrix
         */

        // Number of variables and clusters
        int n_variables = m_u.size();
        int n_clusters = m_R.cols();

        // Set the IDs of variables belonging to m to k
        for (int i = 0; i < n_variables; i++) {
            if (m_u(i) == m) {
                m_u(i) = k;
            }
        }

        // Decrease all IDs that are larger than m by 1
        for (int i = 0; i < n_variables; i++) {
            if (m_u(i) > m) {
                m_u(i) -= 1;
            }
        }

        // Compute weights for weighted mean
        double size_km = static_cast<double>(m_p(k) + m_p(m));
        double w_k = static_cast<double>(m_p(k)) / size_km;
        double w_m = static_cast<double>(m_p(m)) / size_km;

        // Update A
        m_A(k) = w_k * m_A(k) + w_m * m_A(m);

        // Update R
        if (m_p(k) == 1) {
            m_R(k, k) = m_R(m, k);
        }

        if (m_p(m) == 1) {
            m_R(m, m) = m_R(k, m);
        }

        // Update value on the diagonal. Take a weighted average of the two
        // elements on the diagonal. As these elements should also be very
        // similar to the block that is formed by R[k, m], also take the average
        // of the previously obtained value and R[k, m]
        // m_R(k, k) = 0.5 * (w_k * m_R(k, k) + w_m * m_R(m, m)) + 0.5 * m_R(k, m);
        m_R(k, k) = w_k * m_R(k, k) + w_m * m_R(m, m);

        for (int i = 0; i < n_clusters; i++) {
            if (i == k || i == m) continue;

            // Update values in row/column k that are not associated with the
            // diagonal
            double new_val = w_k * m_R(i, k) + w_m * m_R(i, m);
            m_R(i, k) = new_val;
            m_R(k, i) = new_val;
        }

        // Before dropping row/column m, also adjust R*
        m_Rstar.row(k) = m_R.row(k);
        m_Rstar.col(k) = m_R.col(k);

        // Finalize the update of R*
        m_Rstar(k, k) += (m_A(k) - m_R(k, k)) / (m_p(k) + m_p(m));

        // Drop row/column m from R and R* and the kth element from A
        drop_variable_inplace(m_R, m);
        drop_variable_inplace(m_Rstar, m);
        drop_variable_inplace(m_A, m);

        // Update p
        m_p(k) += m_p(m);

        // Move cluster sizes of clusters with index larger than m one position
        // to the left
        for (int i = m; i < n_clusters - 1; i++) {
            m_p(i) = m_p(i + 1);
        }
        m_p.conservativeResize(n_clusters - 1);

        // After A and R have been updated, we can compute the new between
        // cluster distances
        set_distances(W);
    }
};

#endif // VARIABLES_H
