.neg_log_likelihood <- function(S, Theta)
{
    return(-log(det(Theta)) + sum(diag(S %*% Theta)))
}
