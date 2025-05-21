#include <RcppEigen.h>


struct CGGMResult {
    Eigen::MatrixXd R;
    Eigen::VectorXd A;
    Eigen::VectorXi u;
    double lambda;
    double loss;
    int n_clusters;

    CGGMResult(const Eigen::MatrixXd& R, const Eigen::VectorXd& A,
               const Eigen::VectorXi& u, double lambda, double loss) : R(R),
               A(A), u(u), lambda(lambda), loss(loss)
    {
        n_clusters = R.cols();
    }
};


struct Node {
    CGGMResult data;
    Node* next;

    Node(const CGGMResult& data) : data(data), next(nullptr) {}
};


struct LinkedList {
private:
    int size;
    Node* head;
    Node* tail;

public:
    LinkedList() : size(0), head(nullptr), tail(nullptr) {}

    ~LinkedList()
    {
        Node* current = head;

        while (current != nullptr) {
            Node* nextNode = current->next;
            delete current;
            current = nextNode;
        }

        head = nullptr;
        tail = nullptr;
    }

    void insert(const CGGMResult& data)
    {
        // Create a new node with the data
        Node* newNode = new Node(data);

        // If the list is empty, make the new data the head
        if (head == nullptr) {
            head = newNode;
            tail = newNode;
        } else {
            // Let the current tail point to the new node and let the new node
            // be the new tail
            tail->next = newNode;
            tail = newNode;
        }

        // Increase size by 1
        size++;
    }

    int get_size() const
    {
        return size;
    }

    int last_clusters() const
    {
        if (tail == nullptr) return 0;

        return tail->data.n_clusters;
    }

    Rcpp::List convert_to_RcppList()
    {
        int n_results = size;
        int n_variables = tail->data.u.size();

        // Sum of all numbers of clusters
        Node* current = head;
        int sum_n_clusters = 0;
        for (int i = 0; i < n_results; i++) {
            sum_n_clusters += current->data.n_clusters;
            current = current->next;
        }

        // Initialize vector with cluster counts
        Eigen::VectorXi cluster_counts(n_results);

        // Initialize vector with values for lambda
        Eigen::VectorXd lambdas(n_results);

        // Initialize vector with values for the loss function
        Eigen::VectorXd losses(n_results);

        // Initialize matrix with cluster identifiers
        Eigen::MatrixXi clusters(n_variables, n_results);

        // Initialize matrix holding R
        Eigen::MatrixXd R(head->data.R.rows(), sum_n_clusters);
        int R_index = 0;

        // Initialize matrix holding A
        Eigen::MatrixXd A(head->data.R.rows(), n_results);

        // First result
        current = head;

        for (int i = 0; i < n_results; i++) {
            // Add lambda
            lambdas(i) = current->data.lambda;

            // Add loss
            losses(i) = current->data.loss;

            // Add the number of clusters
            cluster_counts(i) = current->data.n_clusters;

            // Add column with cluster IDs
            clusters.col(i) = current->data.u;

            // Add A and R
            for (int j = 0; j < A.rows(); j++) {
                if (j < current->data.n_clusters) {
                    A(j, i) = current->data.A(j);

                    for (int k = 0; k < R.rows(); k++) {
                        if (k < current->data.R.rows()) {
                            R(k, R_index) = current->data.R(k, j);
                        } else {
                            R(k, R_index) = 0;
                        }
                    }

                    R_index++;
                } else {
                    A(j, i) = 0;
                }

            }

            current = current->next;
        }

        return Rcpp::List::create(Rcpp::Named("clusters") = clusters.array() + 1,
                                  Rcpp::Named("R") = R,
                                  Rcpp::Named("A") = A,
                                  Rcpp::Named("lambdas") = lambdas,
                                  Rcpp::Named("losses") = losses,
                                  Rcpp::Named("cluster_counts") = cluster_counts);
    }
};
