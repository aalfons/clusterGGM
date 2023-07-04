#include <RcppEigen.h>
#include "utils.h"


Eigen::VectorXd dropVariable(const Eigen::VectorXd& x, int k)
{
    // Number of rows/columns of X
    int n = x.size();

    // Initialize result
    Eigen::VectorXd result(n - 1);

    for (int i = 0; i < n; i++) {
        if (i == k) {
            continue;
        }

        result(i - (i > k)) = x(i);
    }

    return result;
}


void dropVariableInplace(Eigen::VectorXd& x, int k)
{
    // Number of rows/columns of X
    int n = x.size();

    // Shift elements
    for (int i = k; i < n - 1; i++) {
        x(i) = x(i + 1);
    }

    // Resize
    x.conservativeResize(n - 1);
}


Eigen::MatrixXd dropVariable(const Eigen::MatrixXd& X, int k)
{
    // Number of rows/columns of X
    int n = X.rows();

    // Initialize result
    Eigen::MatrixXd result(n - 1, n - 1);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (j == k || i == k) {
                continue;
            }

            result(i - (i > k), j - (j > k)) = X(i, j);
        }
    }

    return result;
}


void dropVariableInplace(Eigen::MatrixXd& X, int k)
{
    // This can be done more efficiently, now the elements in the bottom right
    // block of the matrix are moved twice: once up and once left, can be
    // changed to up and left simultaneously

    // Number of rows/columns
    int n = X.rows();

    // Shift each with index larger than k one position upwards
    for (int j = 0; j < n; j++) {
        for (int i = k; i < n - 1; i++) {
            X(i, j) = X(i + 1, j);
        }
    }

    // We can either tranpose UWU and do the same or shift each column that is
    // larger than k one position to the left
    for (int j = k; j < n - 1; j++) {
        for (int i = 0; i < n; i++) {
            X(i, j) = X(i, j + 1);
        }
    }

    // Resize
    X.conservativeResize(n - 1, n - 1);
}


void printVector(const Eigen::VectorXd& vec)
{
    Rcpp::Rcout << '[';

    for (int i = 0; i < vec.size() - 1; i++) {
        Rcpp::Rcout << vec(i) << ' ';
    }

    Rcpp::Rcout << vec(vec.size() - 1) << "]\n";
}


void printVector(const Eigen::VectorXi& vec)
{
    Rcpp::Rcout << '[';

    for (int i = 0; i < vec.size() - 1; i++) {
        Rcpp::Rcout << vec(i) << ' ';
    }

    Rcpp::Rcout << vec(vec.size() - 1) << "]\n";
}


void printMatrix(const Eigen::MatrixXd& mat)
{
    Rcpp::Rcout << '[';

    for (int i = 0; i < mat.rows(); i++) {
        if (i != 0) Rcpp::Rcout << ' ';

        for (int j = 0; j < mat.cols() - 1; j++) {
            if (mat(i, j) >= 0) Rcpp::Rcout << ' ';

            Rcpp::Rcout << mat(i, j) << ' ';
        }

        if (i == mat.rows() - 1) Rcpp::Rcout << mat(i, mat.cols() - 1) << "]\n";
        else Rcpp::Rcout << mat(i, mat.cols() - 1) << '\n';
    }
}


std::pair<Eigen::MatrixXd, Eigen::VectorXd>
updateRA(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
         const Eigen::VectorXd& values, int k)
{
    // Initialize result
    Eigen::MatrixXd R_new(R);
    Eigen::VectorXd A_new(A);

    // The updating
    updateRAInplace(R_new, A_new, values, k);

    return std::make_pair(R_new, A_new);
}


void updateRAInplace(Eigen::MatrixXd& R, Eigen::VectorXd& A,
                     const Eigen::VectorXd& values, int k)
{
    // Number of clusters
    int n_clusters = R.cols();

    // The updating
    A(k) += values(0);
    R.col(k) += values.tail(n_clusters);
    R.row(k) += values.tail(n_clusters);
    R(k, k) -= values(1 + k);
}


// [[Rcpp::export]]
Eigen::MatrixXd updateInverse(const Eigen::MatrixXd& inverse, const Eigen::VectorXd& vec, int i)
{
    // Sherman-Morrison with A = inverse, u = e_i, v = vec
    Eigen::VectorXd Au = inverse.col(i);
    Eigen::VectorXd vA = vec.transpose() * inverse;
    Eigen::MatrixXd N = Au * vA.transpose();
    double D = 1.0 / (1.0 + vA(i));

    Eigen::MatrixXd result = inverse - D * N;

    // Sherman-Morrison with A = result, u = vec, v = e_i
    Au.noalias() = result * vec;
    vA = result.row(i);
    N.noalias() = Au * vA.transpose();
    D = 1.0 / (1.0 + Au(i));

    result -= D * N;

    printVector(Au);
    printMatrix(result);

    return result;
}


double partialTrace(const Eigen::MatrixXd& S, const Eigen::VectorXi& u, int k)
{
    // Number of elements on the diagonal
    int P = S.cols();

    // Initialize result
    double result = 0;

    for (int i = 0; i < P; i++) {
        if (u(i) != k) continue;

        result += S(i, i);
    }

    return result;
}


// Function that computes U[, k] * S * U[, k]
double sumSelectedElements(const Eigen::MatrixXd& S, const Eigen::VectorXi& u,
                           const Eigen::VectorXi& p, int k)
{
    // Number of rows/columns in S
    int K = S.cols();

    // Number of nonzero indices
    int nnz = p(k);

    // Vector that holds the indices of where u = k
    Eigen::VectorXi indices(nnz);
    int idx = 0;

    for (int i = 0; i < K; i++) {
        if (u(i) != k) continue;

        // Store index and increment the index of the indices vector by one
        indices(idx) = i;
        idx++;

        if (idx == nnz) break;
    }

    // Initialize result
    double result = 0;

    for (int i = 0; i < nnz; i++) {
        for (int j = 0; j < nnz; j++) {
            result += S(indices(j), indices(i));
        }
    }

    return result;
}


// Function that computes U[, k] * S * U[, -k]
Eigen::VectorXd sumMultipleSelectedElements(const Eigen::MatrixXd& S,
                                            const Eigen::VectorXi& u,
                                            const Eigen::VectorXi& p, int k)
{
    // Preliminaries
    int n_clusters = p.size();
    int n_variables = u.size();

    // Vector that holds sum of all elements before the ith element
    Eigen::VectorXi start_index(n_clusters);
    start_index(0) = 0;
    for (int i = 1; i < n_clusters; i++) {
        start_index(i) = start_index(i - 1) + p(i - 1);
    }

    // Vector with the indices of the variables sorted by cluster
    Eigen::VectorXi indices(n_variables);

    // Fill the vector
    Eigen::VectorXi current_index = Eigen::VectorXi::Zero(n_clusters);
    for (int i = 0; i < n_variables; i++) {
        indices(start_index(u(i)) + current_index(u(i))) = i;
        current_index(u(i))++;
    }

    // Compute U[, k] * S, ignore elements of the result that are not used
    // elsewhere
    std::set<int> ignore_idx;           // This can be done more efficiently with a binary
    for (int i = 0; i < p(k); i++) {    // search, as the indices per cluster are sorted
        ignore_idx.insert(indices(start_index(k) + i));
    }

    Eigen::VectorXd intermediate_result = Eigen::VectorXd::Zero(n_variables);
    for (int i = 0; i < n_variables; i++) {
        if (ignore_idx.find(i) != ignore_idx.end()) continue;

        for (int j = 0; j < p(k); j++) {
            intermediate_result(i) += S(indices(start_index(k) + j), i);
        }
    }

    // Now compute (U[, k] * S) * U[, -k]
    Eigen::VectorXd result = Eigen::VectorXd::Zero(n_clusters - 1);

    for (int i = 0; i < n_clusters; i++) {
        if (i == k) continue;

        for (int j = 0; j < p(i); j++) {
            result(i - (i > k)) += intermediate_result(indices(start_index(i) + j));
        }
    }

    return result;
}


Eigen::MatrixXd computeRStar0Inv(const Eigen::MatrixXd& R,
                                 const Eigen::VectorXd& A,
                                 const Eigen::VectorXi& p, int k)
{
    // Number of rows/columns of R
    int K = R.cols();

    // Drop row/column k from R
    Eigen::MatrixXd result = dropVariable(R, k);

    for (int i = 0; i < K; i++) {
        if (i == k) {
            continue;
        }

        result(i - (i > k), i - (i > k)) += (A(i) - R(i, i)) / p(i);
    }

    result = result.inverse();

    return result;
}


Eigen::MatrixXd computeTheta(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
                             const Eigen::VectorXi& u)
{
    // Preliminaries
    int n_variables = u.size();
    int n_clusters = R.cols();
    Eigen::MatrixXd result(n_variables, n_variables);

    // Fill in R
    for (int j = 0; j < n_variables; j++){
        for (int i = 0; i < n_variables; i++) {
            result(i, j) = R(u(i), u(j));
        }
    }

    // Add diagonal component
    for (int i = 0; i < n_variables; i++) {
        result(i, i) = A(u(i));
    }

    return result;
}


void setEqualToClusterInplace(Eigen::MatrixXd& R, Eigen::VectorXd& A,
                              const Eigen::VectorXi& p, int k, int m)
{
    // Number of clusters
    int n_clusters = R.cols();

    A(k) = A(m);

    if (p(m) < 2) {
        R(m, m) = R(k, m);
    }

    R(k, k) = R(m, m);
    R(k, m) = R(m, m);
    R(m, k) = R(m, m);

    for (int i = 0; i < n_clusters; i++) {
        if (i == k || i == m) continue;

        R(i, k) = R(i, m);
        R(k, i) = R(i, m);
    }
}


void setEqualToClusterMeansInplace(Eigen::MatrixXd& R, Eigen::VectorXd& A,
                                   const Eigen::VectorXi& p, int k, int m)
{
    // Number of clusters
    int n_clusters = R.cols();

    // Compute cluster weights
    double w_k = p(k);
    w_k /= p(k) + p(m);
    double w_m = p(m);
    w_m /= p(k) + p(m);

    // Update A
    double new_value = w_k * A(k) + w_m * A(m);
    A(k) = new_value;
    A(m) = new_value;

    if (p(m) < 2) {
        R(m, m) = R(k, m);
    }

    if (p(k) < 2) {
        R(k, k) = R(k, m);
    }

    new_value = w_k * R(k, k) + w_m * R(m, m);
    R(k, k) = new_value;
    R(k, m) = new_value;
    R(m, k) = new_value;
    R(m, m) = new_value;

    for (int i = 0; i < n_clusters; i++) {
        if (i == k || i == m) continue;

        new_value = w_k * R(i, k) + w_m * R(i, m);
        R(i, k) = new_value;
        R(i, m) = new_value;
        R(k, i) = new_value;
        R(m, i) = new_value;
    }
}
