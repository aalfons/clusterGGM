#include <RcppArmadillo.h>
#include <algorithm>
#include <vector>
#include <map>
#include <set>


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

    int count_sets() {
        // Count the number of disjoint sets (number of distinct roots)
        std::set<int> roots;
        for (int i = 0; i < (int) id.size(); i++) {
            roots.insert(root(i));
        }

        return roots.size();
    }
};


// [[Rcpp::export(.count_clusters)]]
int count_clusters(const arma::imat& E, int n)
{
    /* Find the number of clusters (or disjoint sets) in a graph
     *
     * Inputs:
     * E: matrix of edges, each column containing the indices of the vertices
     *      connected by that edge
     *  n: number of vertices
     *
     * Output:
     * The number of clusters
     */

    // Initialize a disjoint set
    DisjointSet djs(n);

    // Fill the disjoint set
    for (int i = 0; i < (int) E.n_cols; i++) {
        int u = E(0, i);
        int v = E(1, i);
        djs.merge(u, v);
    }

    return djs.count_sets();
}


// [[Rcpp::export(.find_subgraphs)]]
Rcpp::IntegerVector find_subgraphs(const arma::imat& E, int n)
{
    /* Find the disconnected subgraphs within a graph defined by its edges
     *
     * Inputs:
     * E: matrix of edges, each column containing the indices of the vertices
     *      connected by that edge
     * n: number of vertices
     *
     * Output:
     * Vector with subgraph IDs for each vertex
     */
    // Initialize a disjoint set
    DisjointSet djs(n);

    // Fill the disjoint set
    for (int i = 0; i < (int) E.n_cols; i++) {
        int u = E(0, i);
        int v = E(1, i);
        djs.merge(u, v);
    }

    // Initialize vector of cluster IDs
    Rcpp::IntegerVector id(n);

    // The roots are random values, we want consecutive cluster IDs, so we make
    // a map for that
    std::map<int, int> id_dict;

    // Initialize the cluster id
    int c = 0;

    for (int i = 0; i < n; i++) {
        int root = djs.root(i);

        // If the root is not present in the dictionary, add it and give it a
        // new cluster id
        auto it = id_dict.find(root);
        if (it == id_dict.end()) {
            id_dict[root] = c;
            c++;
        }

        // Assign the object the correct id
        id(i) = id_dict[root];
    }

    return id;
}


struct Edge {
    int a;
    int b;
    double w;
};


struct Edges {
    std::vector<Edge> edges;

    Edges(const arma::mat& G)
    {
        // Number of edges
        int n_cols = (int) G.n_cols;
        int n = (n_cols * n_cols - n_cols) >> 1;

        edges.reserve(n);

        for (int j = 0; j < n_cols; j++) {
            for (int i = 0; i < j; i++) {
                edges.push_back({i, j, G(i, j)});
            }
        }
    }

    void sort()
    {
        // Sort edges based on their weight
        std::sort(
            edges.begin(), edges.end(),
            [](const Edge& e1, const Edge& e2) { return e1.w < e2.w; }
        );
    }

    int size() const
    {
        return (int) edges.size();
    }

    int u(int index) const
    {
        return edges[index].a;
    }

    int v(int index) const
    {
        return edges[index].b;
    }
};


// [[Rcpp::export(.find_mst)]]
arma::imat find_mst(const arma::mat& G)
{
    /* Find a minimum spanning tree based on a matrix of distances
     *
     * Inputs:
     * G: matrix containing between-vertex distances
     *
     * Output:
     * matrix containing edges that make up the minimum spanning tree
     */

    // Initialize a disjoint set
    DisjointSet djs(G.n_cols);

    // Gather edges from the graph and sort them based on their weights
    Edges E(G);
    E.sort();

    // Initialize minimum spanning tree as a matrix of integers
    arma::imat mst(2, G.n_cols - 1);
    int mst_index = 0;

    // Apply the remainder of Kruskal's algorithm, adding edges with the
    // smallest weight unless they cause a loop
    for (int i = 0; i < E.size(); i++) {
        if (!djs.connected(E.u(i), E.v(i))) {
            mst(0, mst_index) = E.u(i);
            mst(1, mst_index) = E.v(i);
            mst_index++;

            djs.merge(E.u(i), E.v(i));
        }
    }

    return mst.t();
}
