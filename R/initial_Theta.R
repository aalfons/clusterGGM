.initial_Theta <- function(S)
{
    # Initial estimate for Theta
    Theta = tryCatch(
        {
            # Try to compute the inverse using solve(X)
            inv_S = solve(S)
        },
        error = function(e) {
            # In case of an error (non-invertible matrix), use solve(S + I)
            inv_S = solve(S + diag(nrow(S)))

            # Print warning
            warning(
                "S is singular, Theta is initialized as (S + I)^-1",
                call. = FALSE
            )

            return(inv_S)
        }
    )

    return(Theta)
}
