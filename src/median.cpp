#include <RcppEigen.h>
#include <algorithm>
#include <vector>
#include "norms.h"


double median(const Eigen::VectorXd& vec) {
    // Create a temporary vector to store the data from the Eigen vector
    std::vector<double> tempVec(vec.data(), vec.data() + vec.size());

    // Sort the temporary vector in ascending order
    std::sort(tempVec.begin(), tempVec.end());

    // Calculate the median
    int n = tempVec.size();

    if (n % 2 == 0) {
        return (tempVec[n / 2 - 1] + tempVec[n / 2]) / 2.0;
    } else {
        return tempVec[n / 2];
    }
}


// [[Rcpp::export(.medianDistance)]]
double medianDistance(const Eigen::MatrixXd& Theta)
{
    // Number of cols/rows
    int n = Theta.cols();

    // Initialize vector holding distances
    Eigen::VectorXd dists((n * n - n) >> 1);

    // Compute distances
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < j; i++) {
            int index = ((j * j - j) >> 1) + i;
            dists(index) = std::sqrt(squaredNormTheta(Theta, i, j));
        }
    }

    return median(dists);
}
