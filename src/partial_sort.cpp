#include <RcppEigen.h>
#include <algorithm>


// [[Rcpp::export(.k_largest)]]
Eigen::ArrayXi k_largest(const Eigen::VectorXd& vec, int k)
{
    // Initialize indices vector
    Eigen::ArrayXi indices(vec.size());

    // Fill the vector, let indices start at 1
    for (int i = 0; i < vec.size(); i++) {
        indices(i) = i + 1;
    }

    // Partially sort indices vector based on input vector
    std::partial_sort(
        indices.data(), indices.data() + k, indices.data() + indices.size(),
        [&](int i, int j) { return vec(i - 1) > vec(j - 1); }
    );

    // Resize the vector
    indices.conservativeResize(k);

    return indices;
}
