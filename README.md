# clusterGGM: Sparse Gaussian Graphical Modeling with Variable Clustering

Perform sparse estimation of a Gaussian graphical model (GGM) with node aggregation through variable clustering. Currently, the package implements the clusterpath estimator of the Gaussian graphical model (CGGM) ([Touw, Alfons, Groenen & Wilms, 2025](https://doi.org/10.48550/arXiv.2407.00644)).

More information on this method can be found in the following article:

D.J.W. Touw, A. Alfons, P.J.F. Groenen and I. Wilms (2025). *Clusterpath Gaussian Graphical Modeling*. arXiv:2407.00644. doi: [10.48550/arXiv.2407.00644](https://doi.org/10.48550/arXiv.2407.00644)


## Installation

Package `clusterGGM` can be easily installed from the `R` command line via

```
install.packages("remotes")
remotes::install_github("aalfons/clusterGGM")
```

If you already have package `remotes` installed, you can skip the first line.  Moreover, package `clusterGGM` contains `C++` code that needs to be compiled, so you may need to download and install the [necessary tools for MacOS](https://cran.r-project.org/bin/macosx/tools/) or the [necessary tools for Windows](https://cran.r-project.org/bin/windows/Rtools/).


## Report issues and request features

If you experience any bugs or issues or if you have any suggestions for additional features, please submit an issue via the [*Issues*](https://github.com/aalfons/clusterGGM/issues) tab of this repository.  Please have a look at existing issues first to see if your problem or feature request has already been discussed.


## Contribute to the package

If you want to contribute to the package, you can fork this repository and create a pull request after implementing the desired functionality.


## Ask for help

If you need help using the package, or if you are interested in collaborations related to this project, please get in touch with the [package maintainer](https://personal.eur.nl/alfons/).
