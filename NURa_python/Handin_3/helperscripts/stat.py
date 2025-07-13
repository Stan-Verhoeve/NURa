import numpy as np
from scipy.special import gammainc


def pearson(x: np.ndarray, y: np.ndarray = None):
    """
    Calculate the Pearson correlation coefficient given two arrays `x` and `y`.
    If only `x` is given, calculates the auto-correlation coefficient
    Parameters
    ----------
    x : ndarray
        First array
    y : ndarray
        Second array

    Returns
    -------
    r_xy : float
        Pearson correlation coefficient
    """
    if y is None:
        y = x.copy()
    if not np.shape(x) == np.shape(y):
        raise ValueError(
            f"Shape of `x` and `y` should be the same, but got {np.shape(x)} and {np.shape(y)}"
        )
    xy_mean = np.mean(x * y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.var(x)
    y_var = np.var(y)

    denom_inv = np.sqrt(x_var * y_var) ** (-1)

    r_xy = (xy_mean - x_mean * y_mean) * denom_inv

    return r_xy


def chi2_cdf(x, k):
    """
    ChiSquared cumulative distribution

    Parameters
    ----------
    x : float
        Test value
    k : int
        Degree of freedom

    Returns
    -------
    float
        P(chi2 <- x) given k degrees of freedom
    """
    return gammainc(0.5 * k, 0.5 * x)


def G_test(observed: np.ndarray, expected: np.ndarray, DoF: int = 1) -> float:
    """
    Performs a statistical G-test on INTEGER data

    Parameters
    ----------
    observed : ndarray
        Observed data count. Must be integer
    expected : ndarray
        Expected count based on (binned) model
    DoF : int
        Degrees of freedom of the problem

    Returns
    -------
    G : float
        G test-statistic
    """
    if not np.all(isinstance(x, (int, np.int32, np.int64)) for x in observed):
        raise TypeError("Expects all observed counts to be integer.")

    # G equals zero where data is zero (or rather, smaller than some fraction)
    zeros = observed == 0
    G = 2 * np.sum(
        observed[~zeros] * (np.log(observed[~zeros]) - np.log(expected[~zeros]))
    )

    p_value = 1 - chi2_cdf(G, DoF)

    return G, p_value
