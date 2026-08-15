#include <RcppArmadillo.h>


// [[Rcpp::export(.k_largest)]]
Rcpp::IntegerVector k_largest(const arma::vec& vec, int k)
{
    // Sort indices by descending value, let indices start at 1
    arma::uvec order = arma::sort_index(vec, "descend");

    Rcpp::IntegerVector indices(k);
    for (int i = 0; i < k; i++) {
        indices(i) = order(i) + 1;
    }

    return indices;
}
