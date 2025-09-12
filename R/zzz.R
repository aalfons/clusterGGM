.onUnload <- function (libpath) {
    library.dynam.unload("clusterGGM", libpath)
}
