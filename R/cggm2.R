convertFromPar <- function(par)
{
    # Number of variables
    n_vars = round((-1 + sqrt(8 * length((par)))) / 2)

    # Initialize M
    M = matrix(0, nrow = n_vars, ncol = n_vars)

    # Fill in parameters
    v = par[1:n_vars]
    M[upper.tri(M)] = par[-c(1:n_vars)]
    M = M + t(M)

    return(list("M" = M, "v" = v))
}


convertToPar <- function(M, v)
{
    return(c(v, M[upper.tri(M)]))
}

cggm_loss <- function(par, p, u, S, W, lambda)
{
    R = convertFromPar(par)
    A = R$v
    R = R$M

    return(lossRA(R, A, p, u, S, W, lambda))
}


cggm_gradient <- function(par, p, u, S, W, lambda)
{
    R = convertFromPar(par)
    A = R$v
    R = R$M

    # Initialize gradients
    G_R = matrix(0, nrow = nrow(R), ncol = ncol(R))
    G_A = rep(0, length(A))

    for (k in 1:nrow(R)) {
        R_star_0_inv = computeRStar0Inv(R, A, p, k - 1)
        g = gradient(R, A, p, u, R_star_0_inv, S, W, lambda, k - 1, -1)
        G_R[k, ] = g[-1]
        G_A[k] = g[1]
    }

    # Vectorize gradient
    par_gr = convertToPar(G_R, G_A)

    return(par_gr)
}


cggm2 <- function(S, W, lambdas)
{
    Theta = solve(S)
    R = Theta
    A = diag(Theta)
    u = c(1:nrow(S)) - 1
    p = rep(1, nrow(S))
    par0 = convertToPar(R, A)

    result = list()
    control = list()
    control$maxit = 1000

    for (i in 1:length(lambdas)) {
        # Use optimizer
        res = stats::optim(
            par = par0, fn = cggm_loss, gr = cggm_gradient, p = p, u = u, S = S,
            W = W, lambda = lambdas[i], method = "BFGS", control = control
        )
        sol = convertFromPar(res$par)

        # Add result
        res_i = list()
        res_i$A = sol$v
        res_i$R = sol$M
        res_i$Theta = sol$M + diag(sol$v)
        res_i$lambda = lambdas[i]
        res_i$loss = res$value
        result[[i]] = res_i

        # Set new par0 as warm start
        par0 = convertToPar(sol$M, sol$v)
    }

    result$lambdas = lambdas

    losses = c()
    for (i in 1:length(lambdas)) {
        losses = c(losses, result[[i]]$loss)
    }
    result$losses = losses

    return(result)
}
