#ifndef VARIABLES_H
#define VARIABLES_H

#include <RcppArmadillo.h>
#include "utils.h"


struct Variables {
    /* m_D holds one distance per nonzero of m_W, ordered as the column
     * iterators of m_W visit them, so m_W.col_ptrs[j] indexes the first
     * distance of column j. The distances are dense because assigning an exact
     * zero through an sp_mat iterator deletes the element.
     */
    arma::sp_mat m_W;
    arma::vec m_D;
    arma::mat m_R;
    arma::mat m_Rstar;
    arma::vec m_A;
    arma::ivec m_p;
    arma::ivec m_u;

    Variables(const arma::mat& R, const arma::vec& A,
              const arma::sp_mat& W, const arma::ivec& p,
              const arma::ivec& u)
    {
        // Set attributes
        m_R = R;
        m_A = A;
        m_p = p;
        m_u = u;

        // Compute R*
        m_Rstar = R;
        for (int i = 0; i < (int) R.n_cols; i++) {
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
        int n_clusters = m_R.n_rows;

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
        /* Recompute every distance */

        for (int j = 0; j < (int) m_W.n_cols; j++) {
            arma::uword e = m_W.col_ptrs[j];

            for (auto it = m_W.begin_col(j); it != m_W.end_col(j); ++it, ++e) {
                // Row index
                int i = it.row();

                // Compute distance
                m_D(e) = distance(i, j);
            }
        }
    }

    void set_distances(const arma::sp_mat& W)
    {
        /* Adopt the sparsity pattern of W and fill in the distances.
         *
         * Inputs:
         * W: sparse weight matrix
         */

        // Copy W to get the same sparsity structure
        m_W = W;
        m_D.set_size(m_W.n_nonzero);

        // Set the distances between the clusters for which there is a nonzero
        // weight
        update_all_distances();
    }

    void update_cluster(const arma::vec& values,
                        const arma::vec& E, int k)
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
        for (int j = 0; j < (int) m_W.n_cols; j++) {
            arma::uword e = m_W.col_ptrs[j];

            for (auto it = m_W.begin_col(j); it != m_W.end_col(j); ++it, ++e) {
                // Index
                int i = it.row();

                // If i and j are not equal to k, there is a more efficient
                // approach to updating the weights
                if (i == k || j == k) {
                    m_D(e) = distance(i, j);
                } else {
                    // Compute distance
                    double d_ij = E(e);
                    d_ij += m_p(k) * square(m_R(i, k) - m_R(j, k));
                    m_D(e) = std::sqrt(d_ij);
                }
            }
        }

        // Update R*
        m_Rstar.row(k) = m_R.row(k);
        m_Rstar.col(k) = m_R.col(k);
        m_Rstar(k, k) += (m_A(k) - m_R(k, k)) / m_p(k);
    }

    void fuse_clusters(int k, int m, const arma::sp_mat& W)
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
        int n_variables = m_u.n_elem;
        int n_clusters = m_R.n_cols;

        // Set the IDs of variables belonging to m to k
        for (int i = 0; i < n_variables; i++) {
            if (m_u(i) == m) {
                m_u(i) = k;
            }
        }

        // Decrease all IDs that are larger than m by 1.
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
        // elements on the diagonal.
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

        // Drop the cluster size for m, shifting later cluster sizes down
        m_p.shed_row(m);

        // After A and R have been updated, we can compute the new between
        // cluster distances
        set_distances(W);
    }
};

#endif // VARIABLES_H
