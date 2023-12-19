.expand_vector <- function(vector, max_step_size) {
    # Initialize result and length of the input
    result = c()
    n = length(vector)

    for (i in 1:(n - 1)) {
        # Append value at index i
        result = c(result, vector[i])

        if ((vector[i + 1] - vector[i]) <= max_step_size) next

        # Compute number of inserted elements
        n_elements = floor((vector[i + 1] - vector[i]) / max_step_size)

        # Compute sequence
        elements_ins = seq(vector[i], vector[i + 1],
                           length.out = n_elements + 2)

        # Insert values to ensure step size is smaller than max_step_size
        result = c(result, elements_ins[-c(1, n_elements + 2)])
    }

    # Add the last element as well
    result = c(result, vector[n])

    return(result)
}
