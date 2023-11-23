#include <RcppEigen.h>
#include <algorithm>
#include <vector>
#include <map>


struct DisjointSet {
    std::vector<int> id;
    std::vector<int> sz;

    DisjointSet(int N)
    {
        // Set the id of each object to itself and set the sizes to one
        id.resize(N);
        sz.resize(N);

        for (int i = 0; i < N; i++) {
            id[i] = i;
            sz[i] = 1;
        }
    }

    int root(int i)
    {
        // Ascend through the tree until the root is found and apply path
        // compression on the way up
        while(i != id[i]) {
            id[i] = id[id[i]];
            i = id[i];
        }

        return i;
    }

    bool connected(int p, int q)
    {
        // Check if p and q have the same root
        return root(p) == root(q);
    }

    void merge(int p, int q)
    {
        // Change the parent of the root of p into the root of q
        int i = root(p);
        int j = root(q);

        // Return if the roots are the same
        if (i == j) return;

        // Otherwise link the root of the smaller tree to the root of the larger
        // tree
        if (sz[i] < sz[j]) {
            id[i] = j;
            sz[j] += sz[i];
        } else {
            id[j] = i;
            sz[i] += sz[j];
        }
    }

    int countSets() {
        // Count the number of disjoint sets (number of distinct roots)
        std::set<int> roots;
        for (int i = 0; i < id.size(); i++) {
            roots.insert(root(i));
        }

        return roots.size();
    }
};


// [[Rcpp::export(.count_clusters)]]
int count_clusters(const Eigen::MatrixXi& E, int n)
{
    /* Find the number of clusters (or disjoint sets) in a graph
     *
     * Inputs:
     * E: matrix of edges, each column containing the indices of the vertices
     *      connected by that edge
     *
     * Output:
     * The number of clusters
     */

    // Initialize a disjoint set
    DisjointSet djs(n);

    // Fill the disjoint set
    for (int i = 0; i < E.cols(); i++) {
        int u = E(0, i);
        int v = E(1, i);
        djs.merge(u, v);
    }

    return djs.countSets();
}
