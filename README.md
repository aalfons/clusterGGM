# R package CGGMR2

Package `CGGMR2` implements the Clusterpath estimator of the Gaussian Graphical Model (CGGM). To cite `CGGMR2` in publications, please use:

D.J.W. Touw, A. Alfons, P.J.F. Groenen, and I. Wilms (2025). Clusterpath Gaussian Graphical Modeling. _arXiv preprint arXiv:2407.00644_. doi: [10.48550/arXiv.2407.00644](https://doi.org/10.48550/arXiv.2407.00644).

Note that the package is still in an early development stage and that various improvements are planned in the near future. For issues, please use [Github Issues](https://github.com/djwtouw/CGGMR/issues).

## Contents
- [Installation](#installation)
- [Examples](#examples)

## Installation
`CGGMR2` has the following dependencies:
- `dplyr`
- `mvtnorm`
- `Rcpp`
- `RcppEigen`

To install the latest version of `CCMMR2`, clone the repository, open `CGGMR.Rproj` in RStudio, switch to branch `version-0.2`, and press install in the build panel. Alternatively, use `devtools` to install the package from GitHub via
```R
library("devtools")
install_github("djwtouw/CGGMR", ref = "version-0.2")
```
The package can then be loaded via
```R
library("CGGMR2")
```

## Examples
For detailed examples, see `examples.R`.
