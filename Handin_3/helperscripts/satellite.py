import numpy as np
from .integrate import romberg

# Order for Romberg integration
ORDER = 4
XMIN = 1e-4
XMAX = 5


def n(
    x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> np.ndarray:
    """
    Number density profile of satellite galaxies

    Parameters
    ----------
    x : float | ndarray
        Radius in units of virial radius; x = r / r_virial
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    float | ndarray
        Same type and shape as x. Number density of satellite galaxies
        at given radius x.
    """
    return A * Nsat * ((x / b) ** (a - 3)) * np.exp(-((x / b) ** c))

# Chache parameters to avoid doing the integration multiple times
_cached_params = {"abc": None, "Z": None, "dZ": None}
def _rounded_params(a, b, c, digits=12):
    """Round params to digits"""
    return (round(a, digits), round(b, digits), round(c, digits))

def get_partition_and_derivatives(a, b, c):
    """Get cached parameters"""
    global _cached_params
    abc = _rounded_params(a, b, c)

    if _cached_params["abc"] != abc:
        # Cache miss, recompute
        _cached_params["abc"] = abc
        _cached_params["Z"] = partition(a, b, c)
        _cached_params["dZ"] = dpartition_dparams(a, b, c)

    return _cached_params["Z"], _cached_params["dZ"]

def galaxy_dist(x, Nsat, a, b, c):
    """Non-normalised galaxy dist"""
    return 4 * np.pi * Nsat * x ** (a - 1) * b ** (3 - a) * np.exp(-((x / b) ** c))

def dgalaxy_dparam(x, Nsat, a, b, c, which="a"):
    """Derivative of non-normalised dist wrt its params"""
    if which == "a":
        extra_term = np.log(x / b)
    if which == "b":
        extra_term = (c * (x / b) ** c - (a - 3)) / b
    if which == "c":
        extra_term = -1 * np.log(x / b) * (x / b) ** c

    return galaxy_dist(x, Nsat, a, b, c) * extra_term


def partition(a, b, c):
    """Normalisation partition"""
    integrand = lambda x: galaxy_dist(x, 1, a, b, c)

    return romberg(integrand, (XMIN, XMAX), m=ORDER)


def dpartition_dparams(a, b, c):
    """Partition derivative wrt one of its params"""
    df_da = lambda x: dgalaxy_dparam(x, 1, a, b, c, which="a")
    df_db = lambda x: dgalaxy_dparam(x, 1, a, b, c, which="b")
    df_dc = lambda x: dgalaxy_dparam(x, 1, a, b, c, which="c")

    dpart_da = romberg(df_da, (XMIN, XMAX), m=ORDER)
    dpart_db = romberg(df_db, (XMIN, XMAX), m=ORDER)
    dpart_dc = romberg(df_dc, (XMIN, XMAX), m=ORDER)

    return [dpart_da, dpart_db, dpart_dc]


def model(x, Nsat, a, b, c):
    """Normalised model"""
    Z, _ = get_partition_and_derivatives(a, b, c)
    return galaxy_dist(x, Nsat, a, b, c) / Z


def dmodel_dparam(x, Nsat, a, b, c, which="a"):
    """Model derivative wrt one of its params"""
    Z, dZ = get_partition_and_derivatives(a, b, c)
    if which == "a":
        extra_term = np.log(x / b)
        dZ = dZ[0]
    if which == "b":
        extra_term = (c * (x / b) ** c - (a - 3)) / b
        dZ = dZ[1]
    if which == "c":
        extra_term = -1 * np.log(x / b) * (x / b) ** c
        dZ = dZ[2]

    # Product rule
    return galaxy_dist(x, Nsat, a, b, c) * (extra_term / Z - dZ / Z**2)


def bin_function(func, binedges, use_integral=True):
    """Bin function given binedges"""
    N = len(binedges) - 1
    result = np.zeros(N)
    
    if not use_integral:
        centers = 0.5 * (binedges[1:] + binedges[:-1])
        return func(centers) * np.diff(binedges)
    for i in range(N):
        result[i] = romberg(func, (binedges[i], binedges[i + 1]), m=ORDER)

    return result


def binned_model(binedges, Nsat, a, b, c, use_integral=True):
    """Binned galaxy model"""
    func = lambda x: model(x, Nsat, a, b, c)
    return bin_function(func, binedges, use_integral)


def dmodel_dparams_binned(binedges, Nsat, a, b, c, which, use_integral=True):
    """Binned derivative wrapper"""
    dm_dp = lambda x: dmodel_dparam(x, Nsat, a, b, c, which=which)

    return bin_function(dm_dp, binedges, use_integral)
