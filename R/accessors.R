#' @export
get_Theta <- function(object, ...) UseMethod("get_Theta")


#' @method get_Theta CGGM
#' @export
get_Theta.CGGM <- function(object, index, ...)
{
    Theta = NULL

    if (!is.null(index)) {
        # Throw a warning if the solution index is not valid
        if (index <= 0 || index > object$n) {
            warning("Not a valid index")
        } else {
            # Select R
            R = object$R[[index]]

            # If a nonzero sparsity penalty parameter was used, detect sparsity
            if (object$inputs$lambda_lasso > 0) {
                R = object$R[[index]]
                R[abs(R) < object$inputs$eps_lasso] = 0
            }

            # Compute Theta
            Theta = .compute_Theta(
                as.matrix(R), object$A[[index]], object$clusters[index, ] - 1
            )

            # Row and column names
            rownames(Theta) = rownames(object$inputs$S)
            colnames(Theta) = colnames(object$inputs$S)
        }
    }

    return(Theta)
}


#' @method get_Theta CGGM_refit
#' @export
get_Theta.CGGM_refit <- function(object, index, ...)
{
    Theta = NULL

    if (!is.null(index)) {
        # Throw a warning if the solution index is not valid
        if (index <= 0 || index > object$n) {
            warning("Not a valid index")
        } else {
            # Compute Theta
            Theta = .compute_Theta(
                as.matrix(object$R[[index]]), object$A[[index]],
                object$clusters[index, ] - 1
            )

            # Row and column names
            rownames(Theta) = rownames(object$inputs$S)
            colnames(Theta) = colnames(object$inputs$S)
        }
    }

    return(Theta)
}


#' @method get_Theta CGGM_CV
#' @export
get_Theta.CGGM_CV <- function(object, which = NULL, ...)
{
    if (is.null(which)) {
        if (object$best == "fit") {
            return(get_Theta(object$fit$final, index = object$fit$opt_index))
        } else {
            return(get_Theta(object$refit$final, index = object$refit$opt_index))
        }
    } else if (which == "fit") {
        return(get_Theta(object$fit$final, index = object$fit$opt_index))
    } else if (which == "refit") {
        return(get_Theta(object$refit$final, index = object$refit$opt_index))
    } else {
        return(NULL)
    }
}


#' @export
get_clusters <- function(object, ...) UseMethod("get_clusters")


#' @method get_clusters CGGM
#' @export
get_clusters.CGGM <- function(object, index, ...)
{
    clusters = NULL

    if (!is.null(index)) {
        # Throw a warning if the solution index is not valid
        if (index <= 0 || index > object$n) {
            warning("Not a valid index")
        } else {
            clusters = object$clusters[index, ]
        }
    }

    return(clusters)
}


#' @method get_clusters CGGM_refit
#' @export
get_clusters.CGGM_refit <- function(object, index, ...)
{
    clusters = NULL

    if (!is.null(index)) {
        # Throw a warning if the solution index is not valid
        if (index <= 0 || index > object$n) {
            warning("Not a valid index")
        } else {
            clusters = object$clusters[index, ]
        }
    }

    return(clusters)
}


#' @method get_clusters CGGM_CV
#' @export
get_clusters.CGGM_CV <- function(object, which = NULL, ...)
{
    if (is.null(which)) {
        if (object$best == "fit") {
            return(get_clusters(object$fit$final, index = object$fit$opt_index))
        } else {
            return(get_clusters(object$refit$final, index = object$refit$opt_index))
        }
    } else if (which == "fit") {
        return(get_clusters(object$fit$final, index = object$fit$opt_index))
    } else if (which == "refit") {
        return(get_clusters(object$refit$final, index = object$refit$opt_index))
    } else {
        return(NULL)
    }
}
