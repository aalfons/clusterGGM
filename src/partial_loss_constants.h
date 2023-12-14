#ifndef PARTIALLOSSCONSTANTS_H
#define PARTIALLOSSCONSTANTS_H

#include <RcppEigen.h>
#include "utils.h"
#include "variables.h"


struct PartialLossConstants {
    Eigen::SparseMatrix<double> m_E;
    Eigen::VectorXd m_uSU;
    double m_uSu;
    double m_pTraceS;

    PartialLossConstants(const Variables& vars, const Eigen::MatrixXd& S, int k)
    {
        /* Compute constants for the computation of the partial loss with
         * respect to cluster k
         *
         * Inputs:
         * vars: struct containing the optimization variables
         * S: sample covariance matrix
         * k: cluster of interest
         */

        // Create references to the variables in the struct
        const Eigen::MatrixXd &R = vars.m_R;
        const Eigen::VectorXi &p = vars.m_p;
        const Eigen::VectorXi &u = vars.m_u;

        // Copy the distance matrix, it serves as a starting point for computing
        // distances after modifying one row/column of R
        m_E = vars.m_D;

        for (int j = 0; j < m_E.outerSize(); j++) {
            // Iterator
            Eigen::SparseMatrix<double>::InnerIterator E_it(m_E, j);

            for (; E_it; ++E_it) {
                // In this loop, compute D^2 - p(k) * (R(i, k) - R(j, k)) for
                // all i and j not equal to k. This allows all distances between
                // i and j (again not equal to k) to be calculated much faster,
                // as only 5 additional flops are required to compute the new
                // distance instead of O(n_clusters) flops

                // Index
                int i = E_it.row();

                // Skip iteration
                if (i == k || j == k) continue;

                // Part that has to be subtracted
                double sub = p(k) * square(R(i, k) - R(j, k));
                E_it.valueRef() = square(E_it.value()) - sub;
            }
        }

        // Compute uSU and uSu, which are sums of selected elements in S
        m_uSU = sum_multiple_selected_elements(S, u, p, k);
        m_uSu = sum_selected_elements(S, u, p, k);

        // Compute the trace of S that
        m_pTraceS = partial_trace(S, u, k);
    }
};

#endif // PARTIALLOSSCONSTANTS_H
