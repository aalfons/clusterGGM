rm(list = ls())
gc()
library(CGGMR)

# Generate some covariance data
set.seed(1)
data = generate_covariance(n_vars = 5, n_clusters = 3, n_draws = 1000)

# Tune grid for the cross validation
tune_grid = expand.grid(k = 1:3, phi = seq(0.5, 2.5, 0.5))
tune_grid = expand.grid(k = 1:3, phi = seq(0.5, 2.5, 0.5),
                        lambda = seq(0.01, 1, 0.01))

# Perform cross validation
set.seed(1)
res_cv = cggm_cv2(data$data, tune_grid)

res_cv$opt_tune
res_cv$scores
res_cv$scores[which.min(res_cv$scores$score), ]

get_Theta(res_cv)
get_clusters(res_cv)

set.seed(1)
res_cv_old = cggm_cv(data$data, lambdas = seq(0.01, 1, 0.01),
                     phi = seq(0.5, 2.5, 0.5), k = c(1:3))

res_cv_old$k
res_cv_old$phi
res_cv_old$lambda
min(res_cv_old$scores)
