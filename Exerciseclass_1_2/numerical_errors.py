import numpy as np

# For plotting
import matplotlib.pyplot as plt
from timeit import timeit

def prod(array):
    assert isinstance(array, (list, np.ndarray)), "array should be list or ndarray"
    assert len(array) > 0, "length of array should be greater than 0"

    current = 1
    for el in range(1, len(array)+1):
        current *= el
    return current

def cumprod(array):
    return np.array([prod(array[:i]) for i in range(1, len(array) + 1)])

def factorial(n):
    """
    Calculate the factorial of n using for-loop
    """
    n = np.asarray(n)
    assert np.all(n > 0), "n should be positive"
    largest_n = np.max(n)
    
    return cumprod(np.arange(1, largest_n + 1))[n-1]

def power_series_sinc(x, k):
    assert isinstance(k, int), "k should be an integer"
    assert k > 0, "k should be positive"
    # sinc(0) == 1 by definitionn
    # TODO: make it work


    ks = np.arange(1, k)
    factorials = factorial(2 * ks + 1)
     
    return 1 + np.sum( (-1) ** ks + x ** (2 * ks) / factorials)

print(power_series_sinc(np.array([1,2,3,4,5]), 10))
print(np.sinc(5 / np.pi))
