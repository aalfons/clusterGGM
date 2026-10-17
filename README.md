# clusterGGM: Sparse Gaussian Graphical Modeling with Variable Clustering

Perform sparse estimation of a Gaussian graphical model (GGM) with node aggregation through variable clustering. Currently, the package implements the clusterpath estimator of the Gaussian graphical model (CGGM) ([Touw, Alfons, Groenen & Wilms, 2026](https://doi.org/10.1080/10618600.2026.2653764)) and the tree-aggregated graphical lasso (tag-lasso) ([Wilms & Bien, 2022](https://jmlr.org/papers/v23/21-0105.html)).

More information on this method can be found in the following article:

D.J.W. Touw, A. Alfons, P.J.F. Groenen, and I. Wilms (2026). Clusterpath Gaussian Graphical Modeling. *Journal of Computational and Graphical Statistics*. doi: [10.1080/10618600.2026.2653764](https://doi.org/10.1080/10618600.2026.2653764)

I. Wilms and J. Bien (2022) Tree-based Node Aggregation in Sparse Graphical
Models. *Journal of Machine Learning Research*, 23(243), 1--36.
<https://jmlr.org/papers/v23/21-0105.html>


## Installation

Package `clusterGGM` is on CRAN (The Comprehensive R Archive Network),
hence the latest release can be easily installed from the `R` command
line via

```
install.packages("clusterGGM")
```


## Building from source

To install the latest (possibly unstable) development version from
GitHub, you can pull this repository and install it from the `R` command
line via

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
