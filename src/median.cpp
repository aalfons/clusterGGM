#include <RcppArmadillo.h>
#include "norms.h"


// [[Rcpp::export(.median_distance)]]
double median_distance(const arma::mat& Theta)
{
    // Number of cols/rows
    int n = Theta.n_cols;

    // Initialize vector holding distances
    arma::vec dists((n * n - n) >> 1);

    // Compute distances
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < j; i++) {
            int index = ((j * j - j) >> 1) + i;
            dists(index) = std::sqrt(squared_norm_Theta(Theta, i, j));
        }
    }

    return arma::median(dists);
}
