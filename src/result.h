#ifndef RESULT_H
#define RESULT_H

#include <RcppArmadillo.h>
#include <algorithm>
#include <vector>


struct CGGMResult {
    arma::mat R;
    arma::vec A;
    arma::ivec u;
    double lambda;
    double loss;
    int n_clusters;

    CGGMResult(const arma::mat& R, const arma::vec& A,
               const arma::ivec& u, double lambda, double loss) : R(R),
               A(A), u(u), lambda(lambda), loss(loss)
    {
        n_clusters = R.n_cols;
    }
};


inline Rcpp::List convert_to_RcppList(const std::vector<CGGMResult>& results)
{
    int n_results = (int) results.size();

    if (n_results < 1) {
        Rcpp::stop("no solutions were computed, lambda_cpath is empty");
    }

    int n_variables = results.back().u.n_elem;

    // Sum and maximum of the numbers of clusters
    int sum_n_clusters = 0;
    int max_n_clusters = 0;
    for (int i = 0; i < n_results; i++) {
        sum_n_clusters += results[i].n_clusters;
        max_n_clusters = std::max(max_n_clusters, results[i].n_clusters);
    }

    // Initialize vector with cluster counts.
    Rcpp::IntegerVector cluster_counts(n_results);

    // Initialize vector with values for lambda
    Rcpp::NumericVector lambdas(n_results);

    // Initialize vector with values for the loss function
    Rcpp::NumericVector losses(n_results);

    // Initialize matrix with cluster identifiers
    arma::imat clusters(n_variables, n_results);

    // Initialize matrix holding R
    arma::mat R(max_n_clusters, sum_n_clusters);
    int R_index = 0;

    // Initialize matrix holding A
    arma::mat A(max_n_clusters, n_results);

    for (int i = 0; i < n_results; i++) {
        const CGGMResult& current = results[i];

        // Add lambda
        lambdas(i) = current.lambda;

        // Add loss
        losses(i) = current.loss;

        // Add the number of clusters
        cluster_counts(i) = current.n_clusters;

        // Add column with cluster IDs
        clusters.col(i) = current.u;

        // Add A and R
        for (int j = 0; j < (int) A.n_rows; j++) {
            if (j < current.n_clusters) {
                A(j, i) = current.A(j);

                for (int k = 0; k < (int) R.n_rows; k++) {
                    if (k < (int) current.R.n_rows) {
                        R(k, R_index) = current.R(k, j);
                    } else {
                        R(k, R_index) = 0;
                    }
                }

                R_index++;
            } else {
                A(j, i) = 0;
            }
        }
    }

    return Rcpp::List::create(Rcpp::Named("clusters") = clusters + 1,
                              Rcpp::Named("R") = R,
                              Rcpp::Named("A") = A,
                              Rcpp::Named("lambdas") = lambdas,
                              Rcpp::Named("losses") = losses,
                              Rcpp::Named("cluster_counts") = cluster_counts);
}

#endif // RESULT_H
