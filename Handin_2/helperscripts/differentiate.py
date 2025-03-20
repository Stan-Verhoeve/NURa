import numpy as np


def finite_difference(function: callable, x: np.ndarray, h: float) -> np.ndarray:
    """
    Computes derivative using finite differences

    Parameters
    ----------
    function : callable
        Function to differentiate
    x : float | ndarray
        Value(s) to evaluate derivative at
    h : float
        Step size for finite difference

    Returns
    -------
    dy : float | ndarray
        Derivative at x
    """
    h_inv = h ** (-1)
    dy = (function(x + h) - function(x - h)) * h_inv * 0.5

    return dy


def ridder(function: callable, x: np.ndarray, h_init: float, d: float, eps: float, max_iters: int=10) -> np.ndarray:
    """
    Ridder's method of differentiation

    Parameters
    ----------
    function : callable
        Function to differentiate
    x : float | ndarray
        Value(s) to evaluate derivative at
    h_init : float
        Initial step size for finite difference
    d : float
        Factor by which to decrease h_init every iteration
    eps : float
        Relative error
    max_iters : int
        Maximum number of iterations before exiting

    Returns
    -------
    df : float | ndarray
        Derivative at x
    """

    x = np.atleast_1d(x)
    ridder_matrix = np.empty((max_iters, len(x)))
    d_arr = d**2 * (np.arange(1, max_iters))

    d_inv = d ** (-1)

    # Populate ridder matrix
    for i in range(max_iters):
        ridder_matrix[i] = finite_difference(function, x, h_init * (d_inv**i))

    previous = ridder_matrix[0].copy()
    for j in range(1, max_iters):
        factor = d_arr[j - 1]

        ridder_matrix = (factor * ridder_matrix[1:] - ridder_matrix[:-1]) / (
            factor - 1
        )

        # for i in range(max_iters-j):
        #     ridder_matrix[i] = (d ** (2*j) * ridder_matrix[i+1] - ridder_matrix[i]) / (d**(2*j) - 1)
        error = abs(ridder_matrix[0] - previous)
        previous = ridder_matrix[0].copy()
        if np.all(error < eps):
            break

    if j == max_iters:
        print("Warning: maximum iterations reached!")

    df = ridder_matrix[0]

    # for a single x
    # D[:-1, j] = D[1:, j-1] - D[:-1, j-1]

    # for i in range(np.shape(D)[0]):
    return df
