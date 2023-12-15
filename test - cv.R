rm(list = ls())
gc()
library(CGGMR)

# Generate some covariance data
set.seed(1)
data = generate_covariance(20, 15)

# Compute weight matrix
W = cggm_weights(data$sample, phi = 1, k = 2)

lambda = seq(0, 0.25, 0.05)

# Compute solutions for the initial set of lambdas
result_1 = cggm(S = data$sample, W = W, lambda = lambda, expand = TRUE)

result_1$losses
result_1$lambdas
result_1$cluster_counts


################################################################################

X = data$data
kfold = 5
cov_method = "pearson"
connected = TRUE

# Create folds for k fold cross validation
n = nrow(X)
folds = cv_folds(n, K = kfold)


f_i = 1
# Select training and test samples for fold f_i
X_train = X[-folds[[f_i]], ]
X_test = X[folds[[f_i]], ]
S_train = stats::cov(X_train, method = cov_method)
S_test = stats::cov(X_test, method = cov_method)


phi = 0.5
k = 3

# Run the algorithm on all folds to get a sequence for lambda for this
# combination of k and phi
lambdas = seq(0, 1, 0.1)
S = cov(X)
W = cggm_weights(S, phi, k)
res = cggm(S = S, W = W, lambda = lambdas, expand = TRUE)
res$lambdas

# Compute the weight matrix based on the training sample
W_train = cggm_weights(S_train, phi = phi, k = k, connected = connected)

# Run the algorithm
res = cggm(S = S_train, W = W_train, lambda = lambdas, expand = TRUE)
res$cluster_counts

scores = rep(0, res$n)

for (i in 1:res$n) {
    scores[i] = CGGMR:::.log_likelihood(S_test, get_Theta(res, i))
}
