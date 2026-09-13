# Reconstruct the estimated matrix from the parametrization in terms of A and R.
# This is the inverse of the matrix that was supplied to the estimator, which is
# the precision matrix for cggm() and the covariance matrix for cggm_cov(). Not
# an S3 method, so that it can also be applied to a result before its class has
# been set
.get_estimate <- function(object, index, threshold = TRUE)
{
    estimate = NULL

    if (!is.null(index)) {
        # Throw a warning if the solution index is not valid
        if (index <= 0 || index > object$n) {
            warning("Not a valid index")
        } else {
            R = as.matrix(object$R[[index]])

            # If a nonzero sparsity penalty parameter was used, detect sparsity
            if (threshold && object$inputs$lambda_lasso > 0) {
                R[abs(R) < object$inputs$eps_lasso] = 0
            }

            estimate = .compute_Theta(
                R, object$A[[index]], object$clusters[index, ] - 1
            )

            # Row and column names
            rownames(estimate) = rownames(object$inputs$S)
            colnames(estimate) = colnames(object$inputs$S)
        }
    }

    return(estimate)
}


# Extract the requested matrix, inverting the estimate if the other matrix was
# estimated
.get_target <- function(object, index, target)
{
    estimate = .get_estimate(object, index)

    if (is.null(estimate) || identical(object$matrix, target)) {
        return(estimate)
    }

    return(solve(estimate))
}


# Select the solution of a cross validation result that an accessor function
# should operate on. Returns NULL if which is neither of the two branches
.cv_select <- function(object, which)
{
    if (is.null(which)) which = object$best

    if (identical(which, "fit")) {
        return(list(object = object$fit$final, index = object$fit$opt_index))
    } else if (identical(which, "refit")) {
        return(list(object = object$refit$final, index = object$refit$opt_index))
    }

    return(NULL)
}


#' Extract the Estimated Precision or Covariance Matrix
#'
#' Extract a (block-structured and sparse) precision or covariance matrix
#' obtained via the clusterpath estimator of the Gaussian graphical model
#' (CGGM).
#'
#' The estimator obtains either the precision matrix (see \code{\link{cggm}()})
#' or the covariance matrix (see \code{\link{cggm_cov}()}). Both accessor
#' functions can be applied regardless of which of the two was estimated, as
#' the estimate is inverted when the other matrix is requested. Note that,
#' inverting retains the block structure implied by the cluster assignment.
#'
#' @param object  an object from which to extract the precision or covariance
#' matrix.
#' @param index  an integer specifying the step along the clusterpath for which
#' to extract the precision or covariance matrix.
#' @param which  a character string specifying for which solution to extract
#' the precision or covariance matrix.  Possible values are \code{"refit"} for
#' the solution including the refitting step (see \code{\link{cggm_refit}()}),
#' or \code{"fit"} for the solution without without the refitting step (see
#' \code{\link{cggm}()}).  If \code{NULL} (the default), the solution with the
#' better cross-validation score is used.
#' @param \dots  additional arguments are currently ignored.
#'
#' @return
#' The estimated precision matrix for \code{get_precision()} or the estimated
#' covariance matrix for \code{get_covariance()}.  The matrix is
#' block-structured and sparse if it is the one that was estimated, and
#' block-structured but in general not sparse if it is obtained by inverting the
#' estimate.
#'
#' For the \code{"CGGM_CV"} method (see \code{\link{cggm_cv}()}), the returned
#' matrix corresponds to the optimal values of the tuning parameters.
#'
#' @author Daniel J.W. Touw
#'
#' @references
#' D.J.W. Touw, A. Alfons, P.J.F. Groenen and I. Wilms (2025)
#' \emph{Clusterpath Gaussian Graphical Modeling}. arXiv:2407.00644.
#' doi:10.48550/arXiv.2407.00644.
#'
#' @seealso
#' \code{\link{cggm}()}, \code{\link{cggm_cov}()}, \code{\link{cggm_refit}()},
#' \code{\link{cggm_cv}()}
#'
#' \code{\link{get_clusters}()}
#'
#' @example inst/doc/examples/example-get_precision.R
#'
#' @export
get_precision <- function(object, ...) UseMethod("get_precision")


#' @rdname get_precision
#' @method get_precision CGGM_path
#' @export
get_precision.CGGM_path <- function(object, index, ...)
{
    return(.get_target(object, index, "precision"))
}


#' @rdname get_precision
#' @method get_precision CGGM_CV
#' @export
get_precision.CGGM_CV <- function(object, which = NULL, ...)
{
    selected = .cv_select(object, which)

    if (is.null(selected)) {
        return(NULL)
    }

    return(get_precision(selected$object, index = selected$index))
}


#' @rdname get_precision
#' @export
get_covariance <- function(object, ...) UseMethod("get_covariance")


#' @rdname get_precision
#' @method get_covariance CGGM_path
#' @export
get_covariance.CGGM_path <- function(object, index, ...)
{
    return(.get_target(object, index, "covariance"))
}


#' @rdname get_precision
#' @method get_covariance CGGM_CV
#' @export
get_covariance.CGGM_CV <- function(object, which = NULL, ...)
{
    selected = .cv_select(object, which)

    if (is.null(selected)) {
        return(NULL)
    }

    return(get_covariance(selected$object, index = selected$index))
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
#' \code{\link{cggm}()}, \code{\link{cggm_cov}()}, \code{\link{cggm_refit}()},
#' \code{\link{cggm_cv}()}
#'
#' \code{\link{get_precision}()}, \code{\link{get_covariance}()}
#'
#' @example inst/doc/examples/example-get_clusters.R
#'
#' @export
get_clusters <- function(object, ...) UseMethod("get_clusters")


#' @rdname get_clusters
#' @method get_clusters CGGM_path
#' @export
get_clusters.CGGM_path <- function(object, index, ...)
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
    selected = .cv_select(object, which)

    if (is.null(selected)) {
        return(NULL)
    }

    return(get_clusters(selected$object, index = selected$index))
}
