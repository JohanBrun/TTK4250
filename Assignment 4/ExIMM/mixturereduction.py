from typing import Tuple

import numpy as np


def gaussian_mixture_moments(
    w: np.ndarray,  # the mixture weights shape=(N,)
    mean: np.ndarray,  # the mixture means shape(N, n)
    cov: np.ndarray,  # the mixture covariances shape (N, n, n)
) -> Tuple[
    np.ndarray, np.ndarray
]:  # the mean and covariance of of the mixture shapes ((n,), (n, n))
    """Calculate the first two moments of a Gaussian mixture"""

    # mean
    mean_bar = np.average(mean, axis=0, weights=w)  # TODO: hint np.average using axis and weights argument

    # covariance
    # # internal covariance
    cov_int = np.average(cov, axis=0, weights=w)  # TODO: hint, also an average

    # # spread of means
    # Optional calc: mean_diff =
    mean_diff = mean - mean_bar[None]
    # cov_ext = np.average(mean_diff @ mean_diff.T, axis=0, weights=w)  # TODO: hint, also an average
    cov_ext = np.average(mean_diff[:, :, None] * mean_diff[:, None, :], axis=0, weights=w)

    # # total covariance
    cov_bar = cov_int + cov_ext  # TODO

    return mean_bar, cov_bar
