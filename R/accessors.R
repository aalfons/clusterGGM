#' Extract the Estimated Precision Matrix
#'
#' Extract a (block-structured and sparse) precision matrix obtained via the
#' clusterpath estimator of the Gaussian graphical model (CGGM).
#'
#' @param object  an object from which to extract the precision matrix.
#' @param index  an integer specifying the step along the clusterpath for which
#' to extract the precision matrix.
#' @param which  a character string specifying for which solution to extract
#' the precision matrix.  Possible values are \code{"refit"} for the solution
#' including the refitting step (see \code{\link{cggm_refit}()}), or
#' \code{"fit"} for the solution without without the refitting step (see
#' \code{\link{cggm}()}).  If \code{NULL} (the default), the solution with the
#' better cross-validation score is used.
#' @param \dots  additional arguments are currently ignored.
#'
#' @return
#' The estimated (block-structured and sparse) precision matrix.
#'
#' For the \code{"CGGM_CV"} method (see \code{\link{cggm_cv}()}), the returned
#' precision matrix corresponds to the optimal values of the tuning parameters.
#'
#' @author Daniel J.W. Touw
#'
#' @references
#' D.J.W. Touw, A. Alfons, P.J.F. Groenen and I. Wilms (2025)
#' \emph{Clusterpath Gaussian Graphical Modeling}. arXiv:2407.00644.
#' doi:10.48550/arXiv.2407.00644.
#'
#' @seealso
#' \code{\link{cggm}()}, \code{\link{cggm_refit}()}, \code{\link{cggm_cv}()}
#'
#' \code{\link{get_clusters}()}
#'
#' @example inst/doc/examples/example-get_Theta.R
#'
#' @export
get_Theta <- function(object, ...) UseMethod("get_Theta")


#' @rdname get_Theta
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


#' @rdname get_Theta
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


#' @rdname get_Theta
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


#' Extract the Cluster Assignment
#'
#' Extract a cluster assignment obtained via the clusterpath estimator of the
#' Gaussian graphical model (CGGM).
#'
#' @param object  an object from which to extract the cluster assignment.
#' @param index  an integer specifying the step along the clusterpath for which
#' to extract the cluster assignment.
#' @param which  a character string specifying for which solution to extract
#' the cluster assignment.  Possible values are \code{"refit"} for the solution
#' including the refitting step (see \code{\link{cggm_refit}()}), or
#' \code{"fit"} for the solution without without the refitting step (see
#' \code{\link{cggm}()}).  If \code{NULL} (the default), the solution with the
#' better cross-validation score is used.
#' @param \dots  additional arguments are currently ignored.
#'
#' @return
#' An integer vector giving the obtained cluster assignment for each variable.
#'
#' For the \code{"CGGM_CV"} method (see \code{\link{cggm_cv}()}), the returned
#' cluster assignment corresponds to the optimal values of the tuning
#' parameters.
#'
#' @author Daniel J.W. Touw
#'
#' @references
#' D.J.W. Touw, A. Alfons, P.J.F. Groenen and I. Wilms (2025)
#' \emph{Clusterpath Gaussian Graphical Modeling}. arXiv:2407.00644.
#' doi:10.48550/arXiv.2407.00644.
#'
#' @seealso
#' \code{\link{cggm}()}, \code{\link{cggm_refit}()}, \code{\link{cggm_cv}()}
#'
#' \code{\link{get_Theta}()}
#'
#' @example inst/doc/examples/example-get_clusters.R
#'
#' @export
get_clusters <- function(object, ...) UseMethod("get_clusters")


#' @rdname get_clusters
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


#' @rdname get_clusters
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


#' @rdname get_clusters
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
