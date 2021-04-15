## usethis namespace: start
#' @useDynLib clusterglasso, .registration = TRUE
## usethis namespace: end
NULL

## usethis namespace: start
#' @importFrom Rcpp sourceCpp
## usethis namespace: end
NULL


## usethis namespace: start
#' @exportPattern "^[[:alpha:]]+"
## usethis namespace: end
NULL

.onUnload <- function (libpath) {
  library.dynam.unload("clusterglasso", libpath)
}
