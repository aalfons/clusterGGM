#include <RcppArmadillo.h>
#include "norms.h"
#include "utils.h"


double squared_norm_Theta(const arma::mat& Theta, int i, int j)
{
    // Number of rows/columns of theta
    int K = Theta.n_cols;

    // Initialize result
    double result = square(Theta(i, i) - Theta(j, j));

    // Fill result
    for (int k = 0; k < K; k++) {
        if (k == i || k == j) {
            continue;
        }
        result += square(Theta(k, i) - Theta(k, j));
    }
    return result;
}
