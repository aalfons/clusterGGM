.log_likelihood <- function(S, Theta)
{
    log(det(Theta)) - sum(diag(S %*% Theta))
}
