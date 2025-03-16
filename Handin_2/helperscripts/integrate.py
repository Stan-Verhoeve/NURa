import numpy as np
import warnings

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
    USE_OPEN = False
    if not callable(func):
        raise TypeError(
            "Expected 'func' to be callable, but got {type(func).__name__}"
        )

    # Extract bounds and first step size
    lower, upper = bounds
    h = upper - lower
    

    # TODO: Check if this is correct
    # Closed integration
    dividend = 2
    try:
        __ = func(lower, *args), func(upper, *args)
    except:
        # warnings.warn("Function cannot be evaluated at (one of) the bounds. Switching to midpoint instead", category=UserWarning)
        print("Function cannot be evaluated at (one of) the bounds. Switching to midpoint instead.")
        USE_OPEN = True
        dividend = 3

        # Shift h to be the length to first midpoint
        h /= 2
    
    # Array to hold integral guesses
    r = np.zeros(m)
    
    # TODO: check if open integration is correct

    if USE_OPEN:
        # Initial midpoint (on full domain)
        r[0] = 2 * h * func(lower + h, *args)
    else:
        # Initial trapezoid (on full domain)
        r[0] = 0.5 * h * (func(lower, *args) + func(upper, *args))
    
    # TODO: check if correct
    # Number of points in refinement
    N = 1

    # Multiplier for new estimate. Since shifted h --> h/2 in case
    # of open integration, we should convert this back when calculating
    # the new estimate
    multiplier = (dividend - 1)
    for i in range(1, m):
        Delta = h
        h /= dividend
        N *= dividend
        
        # New points to evaluate
        if USE_OPEN:
            # Grab lower + odd h, but not every 3rd odd number
            # since these have been calculated in previous loop already
            newx = np.array([lower + (2 * k + 1) * h for k in range(N) if (k % 3) != 1])
        else:
            newx = np.arange(lower + h, upper, Delta)
        newy = func(newx, *args)

        # New estimate
        r[i] = (r[i - 1] + (dividend - 1) * Delta * np.sum(newy)) / dividend

    # Iteratively improve solution
    factor = 1.0
    for i in range(m):
        factor *= (dividend ** 2)
        for j in range(1, m - i):
            r[j - 1] = (factor * r[j] - r[j - 1]) / (factor - 1)

    if err:
        return r[0], abs(r[0] - r[1])
    return r[0]
