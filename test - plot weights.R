# Generate covariance matrix
#set.seed(1)
data = generateCovariance(n_vars = 5, n_clusters = 2)
Sigma = data$true
S = data$sample

# Number of knn
k = 2

# Compute weight matrix
W_full = cggm_weights(S, phi = 1, k = k)


Theta = CGGMR:::.initialTheta(S)
D = CGGMR:::.scaled_squared_norms(Theta)^0.5

W_dc = cggm_weights(S, phi = 1, k = k)
W_cc = cggm_weights(S, phi = 1, k = k, connected = TRUE)

C = cmdscale(D, k = 2)

plot(C[, 1], C[, 2], asp = 1, axes = FALSE, xlab = "", ylab = "")

n <- nrow(C)

# Draw lines based on W_dc values
for (i in 1:n) {
    for (j in 1:n) {
        if (i <= j) next
        if (W_dc[i, j] > 0) {
            lines(c(C[i, 1], C[j, 1]), c(C[i, 2], C[j, 2]), col = "red", lty = "dashed")

            # Calculate angle between two points
            angle = atan2(C[j, 2] - C[i, 2], C[j, 1] - C[i, 1]) * 180 / pi

            # Adjust angle if necessary to prevent upside down labels
            if (angle < -90 || angle > 90) {
                angle = angle + 180
            }

            # Add the value of W_dc to the lines
            text(mean(c(C[i, 1], C[j, 1])),
                 mean(c(C[i, 2], C[j, 2])),
                 labels = round(W_dc[i, j], 3),
                 col = "black", srt = angle, adj = c(1, 1))
        }
    }
}

# Plot the coordinates on top
points(C[, 1], C[, 2], pch = 16, col = "blue")

# Add labels to points ranging from 1 to n
text(C[, 1], C[, 2], labels = 1:n, pos = 3, offset = 0.5, col = "black")

title(main = "Weight matrix structure")
