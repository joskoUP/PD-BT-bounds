# Posterior error bounds for prior-driven balancing in linear Gaussian inverse problems

This MATLAB repository contains code for the numerical results of the following paper:

1. König, J., Lie, H. C. "[Posterior error bounds for prior-driven balancing in linear Gaussian inverse problems](put arXiv link here)."

## Summary
The work in [1] proposes error bounds for the posterior mean and covariance approximation in linear Gaussian inverse problems, in which the forward model is reduced via prior-driven balanced truncation (PD-BT) [2]. Additionally, we include theoretical bounds and code for the time-limited version of PD-BT, comparing both bounds to the absolute errors in the numerical example given in this repository.
This work uses code from [2] (to be found at [https://github.com/joskoUP/PD-BT/](https://github.com/joskoUP/PD-BT/)).

## Examples
To run this code, you need the MATLAB Control System Toolbox.

To generate Figure 1 from the paper, run the *ISS_PDBT_both_bounds.m* script.

## References
2. König, J., Qian, E., Freitag, M. A. "[Dimension and model reduction approaches for linear Bayesian inverse problems with rank-deficient prior covariances](http://arxiv.org/abs/2506.23892)."

### Contact
Please feel free to contact [Josie König](https://www.math.uni-potsdam.de/professuren/datenassimilation/personen/josie-koenig) with any questions about this repository or the associated paper.
