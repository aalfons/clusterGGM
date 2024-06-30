# CGGMR

CGGMR implements the Clusterpath estimator of the Gaussian Graphical Model (CGGM). To cite `CGGMR` in publications, please use:

D.J.W. Touw, A. Alfons, P.J.F. Groenen, and I. Wilms (2024). Clusterpath Gaussian Graphical Modeling. _arXiv preprint arXiv:[...]_. doi: https://doi.org/[...].

For issues, please use [Github Issues](https://github.com/djwtouw/CGGMR/issues).

## Contents
- [Installation](#installation)
- [Examples](#examples)

## Installation
CGGMR has the following dependencies:
- dplyr
- mvtnorm
- Rcpp
- RcppEigen

To install CCMMR, clone the repository, open `CGGMR.Rproj` in RStudio, and press install in the build panel. Alternatively, use devtools to install the package from GitHub via
```R
library(devtools)
install_github("djwtouw/CGGMR")
```

## Examples
For detailed examples, see `examples.R`.
