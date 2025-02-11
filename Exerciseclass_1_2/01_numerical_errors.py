import numpy as np

# For plotting
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

def prod(array: list) -> float:
    """
    Product of all elements in an array

    Parameters
    ----------
    array : list | ndarray
        Input array to calculate product of

    Returns
    -------
    product : float
        Product of all elements in `array`
    """
    assert isinstance(array, (list, np.ndarray)), "array should be list or ndarray"
    assert len(array) > 0, "length of array should be greater than 0"
    
    # Product, by definition, starts at 1
    product = 1

    # Iterate over all elements in array
    for el in array:
        product *= el
    return product

def cumprod(array: list) -> list:
    """
    Cumulative product of all elements in array

    Parameters
    ----------
    array : list | ndarray
        Imput array to calculate cumulative product of

    Returns
    -------
    cum : list | ndarray
        Cumulative product of all elements in aray
    """
    
    # Iterate over length of array, add product of array up to i
    cum = np.array([prod(array[:i]) for i in range(1, len(array) + 1)])
    return cum

def factorial(n: int | list[int]) -> int | list[int]:
    """
    Calculate the factorial of n

    Parameters
    ----------
    n : int | list[int]
        Number to calculate factorial of

    Returns
    -------
    int | list[int]
        Factorial of n. If n is a list, return 
        list with factorials of all n in list
    """
    n = np.asarray(n)
    assert np.all(n > 0), "n should be positive"
    largest_n = np.max(n)
    
    return cumprod(np.arange(1, largest_n + 1))[n-1]

def power_series_sinc(x: float, k: int) -> float:
    """
    Calculate sinc(x) based on its power (Maclaurin) series up truncated
    at order `k`.

    Parameters
    ----------
    x : float
        Point at which to evaluate sinc
    k : int
        Truncation order of Maclaurin series

    Returns
    -------
    float
        sinc(x) evaluated at `x` up to order `k`
              
    """
    assert isinstance(k, (int, np.int32, np.int64)), "k should be an integer"
    assert k > 1, "Truncation order should be at least 2"
    
    # TODO: Broadcast to work for arrays of x

    # Values of k. Start at 1, since k=0 results in 1
    # in the Maclaurin series
    ks = np.arange(1, k)

    # Calculate factorials
    factorials = factorial(2 * ks + 1)
    
    # Return Maclaurin series
    return 1 + np.sum( (-1) ** ks * x ** (2 * ks) / factorials)


def main():
    x = 7

    # Divide by pi, since numpy defines sinc as sin(pi x) / (pi x)
    libsinc = np.sinc(x / np.pi)

    # Different truncation orders of k to consider
    truncations = np.arange(2, 11, dtype=int)
    truncated_sinc = np.zeros(truncations.size, dtype=float)
    
    # Iterate over truncation orders, save to array
    for i, k in enumerate(truncations):
        truncated_sinc[i] = power_series_sinc(x, k)

    
    # Difference from sinc defined through library function
    Difference = truncated_sinc - libsinc

    # Relative error in %
    rel_err = 100 * (truncated_sinc - libsinc) * libsinc ** (-1)

    # Plot result
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)

    ax.scatter(truncations, Difference)
    ax.axhline(0, c="k")
    ax.set_xlabel("Truncation k")
    ax.set_ylabel("Difference")

    plt.tight_layout()
    plt.show()

# Run if not imported
if __name__ in ("__main__"):
    main()
