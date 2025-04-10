import numpy as np
from .integrate import romberg


#################################
## Number density distribution ##
#################################
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


def dn_dx(x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float):
    """
    Derivative of number density provide

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
        Same type and shape as x. Derivative of number density of
        satellite galaxies at given radius x.
    """
    return (
        -A
        * Nsat
        * b**3
        * (x / b) ** (a)
        * (c * (x / b) ** c - a + 3)
        * np.exp(-((x / b) ** c))
        / x**4
    )


###########################
## Model N(x) = 4pi n(x) ##
###########################


def model(x, a, b, c):
    ig = lambda x, *args: x**2 * n(x, 1, Ntest, a, b, c)
    I = romberg(ig, (1e-4, 5), m=10, args=(a, c, b))
    # norm = 1/(4*np.pi*I)
    norm = 1
    return 4 * np.pi * x**2 * n(x, norm, Ntest, a, b, c)


######################################
## Model derivatives wrt parameters ##
######################################
def dmodel_da(x, a, b, c):
    return model(x, a, b, c) * np.log(x / b)


def dmodel_db(x, a, b, c):
    ig = lambda x, *args: x**2 * n(x, 1, Ntest, a, b, c)
    I = romberg(ig, (1e-4, 5), m=10, args=(a, c, b))
    # norm = 1/(4*np.pi * I)
    norm = 1
    base = x / b
    power = a - 3
    exp_term = np.exp(-(base**c))
    term1 = -power * base ** (power) / b
    term2 = -c * base ** (power + c) / b
    return 4 * np.pi * x**2 * Ntest * norm * (term1 + term2) * exp_term
    # return model(x, a, c, b) * (c * (x/b)**c - (a-3)) / b


def dmodel_dc(x, a, b, c):
    ig = lambda x, *args: x**2 * n(x, 1, Ntest, a, b, c)
    I = romberg(ig, (1e-4, 5), m=10, args=(a, c, b))
    # norm = 1/(4*np.pi * I)
    norm = 1
    base = x / b
    # return 4 * np.pi * x**2 * Ntest * norm * base**(a - 3 + c) * (-np.log(base)) * np.exp(-base**c)
    return -1 * model(x, a, b, c) * np.log(x / b) * (x / b) ** c
