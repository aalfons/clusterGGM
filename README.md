# CGGMR

CGGMR implements the Clusterpath estimator of the Gaussian Graphical Model (CGGM). To cite `CGGMR` in publications, please use:

D.J.W. Touw, A. Alfons, P.J.F. Groenen, and I. Wilms (2024). Clusterpath Gaussian Graphical Modeling. _arXiv preprint arXiv:2407.00644_. doi: https://doi.org/10.48550/arXiv.2407.00644.

Note that the package is still in an early development stage and that various improvements are planned in the near future. For issues, please use [Github Issues](https://github.com/djwtouw/CGGMR/issues).

## Contents
- [Installation](#installation)
- [Examples](#examples)

## Installation
CGGMR has the following dependencies:
- dplyr
- mvtnorm
- parallel
- Rcpp
- RcppEigen
- stats

To install CGGMR, clone the repository, open `CGGMR.Rproj` in RStudio, and press install in the build panel. Alternatively, use devtools to install the package from GitHub via
```R
library(devtools)
install_github("djwtouw/CGGMR")
```

## Examples
For detailed examples, see `examples.R`.
