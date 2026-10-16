"""Weighted density estimation and likelihood-weighted acquisition.

Adapted from biancach/LikelihoodWeightedActiveSampling; see NOTICE.md.
"""
import numpy as np
from KDEpy import FFTKDE
from scipy.stats import gaussian_kde
from scipy.interpolate import InterpolatedUnivariateSpline


def kde(values, weights=None, bandwidth=None):
    values = np.asarray(values).flatten()
    if bandwidth is None:
        try:
            bandwidth = float(np.sqrt(gaussian_kde(values, weights=weights).covariance).item())
        except (ValueError, np.linalg.LinAlgError):
            bandwidth = 1.
        if bandwidth < 1e-8:
            bandwidth = 1.
    return FFTKDE(bw=bandwidth).fit(values, weights)


def evaluate_density(estimator, grid):
    low = min(grid[0], float(estimator.data.min()) - 3 * float(estimator.bw))
    high = max(grid[-1], float(estimator.data.max()) + 3 * float(estimator.bw))
    covering = np.linspace(low, high, max(4096, len(grid)))
    return np.maximum(np.interp(grid, covering, estimator.evaluate(covering)), 0.)


def acquisition(mean, variance, weights, input_density):
    # Integration masses already encode the prior; do not multiply KDE weights
    # by input_density again. The latter is a density in Cartesian coordinates.
    estimator = kde(mean, weights=weights)
    grid, density = estimator.evaluate()
    output_density = InterpolatedUnivariateSpline(grid, density, k=1)(mean)
    output_density = np.maximum(output_density, np.finfo(float).eps)
    return variance * input_density / output_density


def log_pdf_error(prediction, reference, grid):
    floor = np.exp(-30.)
    difference = abs(np.log(np.maximum(prediction, floor)) - np.log(np.maximum(reference, floor)))
    return np.trapz(difference, grid)
