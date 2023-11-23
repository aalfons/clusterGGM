.convert_to_sparse <- function(W)
{
    # Numer of nonzero elements
    nnz = 2 * sum(W[lower.tri(W)] > 0)

    # Keys and values
    W_keys = matrix(nrow = 2, ncol = nnz)
    W_values = rep(0, nnz)

    # Fill keys and values
    idx = 1
    for (j in 1:ncol(W)) {
        for (i in 1:nrow(W)) {
            if (W[i, j] <= 0) next

            # Fill in keys and values
            W_keys[1, idx] = i - 1
            W_keys[2, idx] = j - 1
            W_values[idx] = W[i, j]
            idx = idx + 1
        }
    }

    # Prepare result
    result = list()
    result$keys = W_keys
    result$values = W_values

    return(result)
}
