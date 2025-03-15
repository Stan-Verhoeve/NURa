import numpy as np


def romberg(
    func: callable, bounds: tuple, m: int = 5, err: bool = False, args: tuple = ()
) -> float:
    """
    Romberg integration method

    Parameters
    ----------
    func : callable
        Function to integrate.
    bounds : tuple
        Lower- and upper bound for integration.
    order : int, optional
        Order of the integration.
        The default is 5.
    err : bool, optional
        Whether to retun first error estimate.
        The default is False.
    args : tuple, optional
        Arguments to be passed to func.
        The default is ().

    Returns
    -------
    float
        Value of the integral. If err=True, returns the tuple
        (value, err), with err a first estimate of the (relative)
        error.
    """
    if not callable(func):
        raise TypeError(
            "Expected 'func' to be callable, but got {type(func).__name__}"
        )

    # Extract bounds and first step size
    lower, upper = bounds
    h = upper - lower

    # Array to hold integral guesses
    r = np.zeros(m)

    # Initial trapezoid (on full domain)
    r[0] = 0.5 * h * (func(lower, *args) + func(upper, *args))

    for i in range(1, m):
        Delta = h
        h *= 0.5

        # New points to evaluate
        newx = np.arange(lower + h, upper, Delta)
        newy = func(newx, *args)

        # New estimate
        r[i] = 0.5 * (r[i - 1] + Delta * np.sum(newy))

    # Iteratively improve solution
    factor = 1.0
    for i in range(m):
        factor *= 4
        for j in range(1, m - i):
            r[j - 1] = (factor * r[j] - r[j - 1]) / (factor - 1)

    if err:
        return r[0], abs(r[0] - r[1])
    return r[0]
