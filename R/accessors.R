get_Theta <- function(object, ...) UseMethod("get_Theta")
get_clusters <- function(object, ...) UseMethod("get_clusters")


get_Theta.CGGM <- function(object, index, ...)
{
    Theta = NULL

    if (!is.null(index)) {
        # Throw a warning if the solution index is not valid
        if (index <= 0 || index > object$n) {
            warning("Not a valid index")
        } else {
            Theta = object$Theta[[index]]
        }
    }

    return(Theta)
}


get_Theta.CGGM_CV <- function(object, ...)
{
    return(get_Theta(object$final, index = object$opt_index))
}


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


get_clusters.CGGM_CV <- function(object, ...)
{
    return(get_clusters(object$final, index = object$opt_index))
}
