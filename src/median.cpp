#include <RcppEigen.h>
#include <algorithm>
#include <vector>
#include "norms.h"


double median(const Eigen::VectorXd& vec) {
    // Create a temporary vector to store the data from the Eigen vector
    std::vector<double> temp_vec(vec.data(), vec.data() + vec.size());

    // Sort the temporary vector in ascending order
    std::sort(temp_vec.begin(), temp_vec.end());

    // Calculate the median
    int n = temp_vec.size();

    if (n % 2 == 0) {
        return (temp_vec[n / 2 - 1] + temp_vec[n / 2]) / 2.0;
    } else {
        return temp_vec[n / 2];
    }
}


// [[Rcpp::export(.median_distance)]]
double median_distance(const Eigen::MatrixXd& Theta)
{
    // Number of cols/rows
    int n = Theta.cols();

    // Initialize vector holding distances
    Eigen::VectorXd dists((n * n - n) >> 1);

    // Compute distances
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < j; i++) {
            int index = ((j * j - j) >> 1) + i;
            dists(index) = std::sqrt(squared_norm_Theta(Theta, i, j));
        }
    }

    return median(dists);
}
