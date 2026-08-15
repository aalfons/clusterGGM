#ifndef PARTIALLOSSCONSTANTS_H
#define PARTIALLOSSCONSTANTS_H

#include <RcppArmadillo.h>
#include "utils.h"
#include "variables.h"


struct PartialLossConstants {
    // Ordered like Variables::m_D
    arma::vec m_E;
    arma::vec m_uSU;
    double m_uSu;
    double m_pTraceS;

    PartialLossConstants(const Variables& vars, const arma::mat& S, int k)
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
        const arma::mat &R = vars.m_R;
        const arma::ivec &p = vars.m_p;
        const arma::ivec &u = vars.m_u;
        const arma::sp_mat &W = vars.m_W;

        // Copy the distances, they serve as a starting point for computing
        // distances after modifying one row/column of R
        m_E = vars.m_D;

        for (int j = 0; j < (int) W.n_cols; j++) {
            arma::uword e = W.col_ptrs[j];

            for (auto W_it = W.begin_col(j); W_it != W.end_col(j); ++W_it, ++e) {
                // In this loop, compute D^2 - p(k) * (R(i, k) - R(j, k)) for
                // all i and j not equal to k. This allows all distances between
                // i and j (again not equal to k) to be calculated much faster,
                // as only 5 additional flops are required to compute the new
                // distance instead of O(n_clusters) flops

                // Index
                int i = W_it.row();

                // Skip iteration
                if (i == k || j == k) continue;

                // Part that has to be subtracted
                double sub = p(k) * square(R(i, k) - R(j, k));
                m_E(e) = square(m_E(e)) - sub;
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
