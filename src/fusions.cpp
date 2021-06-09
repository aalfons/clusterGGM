#include <RcppArmadillo.h>


// Determine which rows are close enough to be fused
arma::uvec assignClusters(const arma::mat& M, const arma::mat& UWU, double fuse_threshold) {
  unsigned int n = M.n_rows;
  unsigned int c = 1;
  double d_ij = 0;
  double fuse_threshold_sq = fuse_threshold * fuse_threshold;
  arma::uvec cluster_vec(n, arma::fill::zeros);

  bool find_cluster_buddies = false;

  /* Loop over each column of the sparse matrix UWU and then over the elements
   * within that column to find candidates for fusions.*/
  for (unsigned int j = 0; j < n; j++) {
    if (cluster_vec(j) == 0) {
      cluster_vec(j) = c;
      find_cluster_buddies = true;
      c++;
    }

    for (unsigned int i = 0; i < n; i++) {
      if (cluster_vec(i) == 0 && find_cluster_buddies && i > j && UWU(i, j) > 0) {
        d_ij = arma::dot(M.row(i) - M.row(j), M.row(i) - M.row(j));

        if (d_ij <= fuse_threshold_sq) {
          cluster_vec.at(i) = cluster_vec.at(j);
        }
      }
    }

    find_cluster_buddies = false;
  }

  return cluster_vec;
}



// Fuses rows of M that are sufficiently close to each other
void find_fusions(arma::mat& M, const arma::mat& UWU, arma::sp_mat& U, double fuse_threshold) {
  int p = M.n_cols;
  int c_old = M.n_rows;
  int n = U.n_rows;

  // Determine cluster assignments and new number of clusters
  arma::uvec cluster_vec = assignClusters(M, UWU, fuse_threshold);
  int c_new = arma::max(cluster_vec);

  // Initialize update matrices U and M
  arma::mat M_new(c_new, p, arma::fill::zeros);
  arma::sp_mat U_update(c_old, c_new);
  arma::sp_mat U_new(n, c_new);

  // Compute the old cluster sizes
  arma::vec sizes_old = arma::vec(arma::sum(U, 0).t());

  // Compute the new matrix U and begin working on the new M
  for (int i=0; i<c_old; i++) {
    U_update(i, cluster_vec.at(i) - 1) = 1;

    for (int j=0; j<p; j++) {
      M_new(cluster_vec(i) - 1, j) += M.at(i, j) * sizes_old.at(i);
    }
  }
  U_new = U * U_update;

  // Compute the new cluster sizes
  arma::vec sizes_new = arma::vec(arma::sum(U_new, 0).t());

  // Divide the new M's rows by the new cluster sizes
  for (int i = 0; i < c_new; i++) {
    for (int j = 0; j < p; j++) {
      M_new.at(i, j) /= sizes_new.at(i);
    }
  }

  M = M_new;
  U = U_new;
}
