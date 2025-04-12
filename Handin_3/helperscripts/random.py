import numpy as np
import time
import os


def get_time_based_seed():
    """
    Returns a seed based on time in microsec
    """
    return np.uint64(time.time() * 1_000_000)


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


class Random:
    def __init__(self, seed=None):
        if seed is None:
            seed = get_time_based_seed()

        seed = np.uint64(seed)

        # These are private; user should not acces them
        self.__xor_state = seed
        self.__mwc_state = seed
        self.__mwc_a = np.uint64(4294957665)

        # Pre-define to avoid overhead
        self._mask32 = np.uint64(2**32 - 1)
        self._2_to_32 = np.uint64(2**32)
        self._32 = np.uint64(32)

    def __next_xorshift(self):
        self.__xor_state ^= self.__xor_state >> np.uint64(13)
        self.__xor_state ^= self.__xor_state << np.uint64(17)
        self.__xor_state ^= self.__xor_state >> np.uint(5)

    def __next_mwc(self):
        self.__mwc_state = self.__mwc_a * (self.__mwc_state & self._mask32) + (
            self.__mwc_state >> self._32
        )

    def _next(self):
        self.__next_xorshift()
        self.__next_mwc()

        return (self.__xor_state ^ self.__mwc_state) & self._mask32

    def _ensure_array(self, value, size):
        """
        Helper function to ensure that `value` is a numpy array that can be broadcasted
        to the specified size. If it's a scalar, it's turned into an array of the same size.
        If it's a list/tuple, it's turned into an array and then broadcasted.

        Parameters
        ----------
        value : scalar, tuple, or list
            The value to convert and broadcast.
        size : tuple
            The desired shape of the output array.

        Returns
        -------
        np.ndarray
            A numpy array of the appropriate shape.
        """
        if isinstance(value, (tuple, list)):
            value = np.array(value)

        return np.broadcast_to(value, size)

    def uniform(self, low: float = 0, high: float = 1, size: int = 1) -> np.ndarray:
        """
        Generate array of uniformly distributed numbers in the range [low, high)

        Parameters
        ----------
        low : float
            Lower end of domain
        high : float
            Higher end of domain
        size : int
            Size of array

        Returns
        -------
        arr : ndarray
            Array containing pseudo-random uniformly distributed numbers
        """
        low = self._ensure_array(low, size)
        high = self._ensure_array(high, size)

        raw = np.empty(size, dtype=np.uint32)
        for idx in np.ndindex(size):
            raw[idx] = self._next()

        # Normalise to U[0,1)
        norm = raw.astype(np.float64) / np.float64(self._2_to_32)

        # Scale default [0, 1) to [low, high)
        arr = norm * (high - low) + low

        return arr

    def randint(self, low=0, high=10, size=1):
        """
        Generate array of uniformly distributed integers in the range [low, high)

        Parameters
        ----------
        low : int
            Lower end of domain
        high : int
            Higher end of domain
        size : int
            Size of array

        Returns
        -------
        arr : ndarray
            Array containing pseudo-random uniformly distributed numbers
        """
        return self.uniform(low, high, size).astype(np.int32)

    def normal(self, mean, std, size=1):
        # Store result
        result = np.zeros(size)

        # If size is odd, add one more sample to generate
        odd = size % 2
        Nsamples = size + odd

        # Uniform samples
        U1 = self.uniform(0, 1, Nsamples // 2)
        U2 = self.uniform(0, 1, Nsamples // 2)

        # Normal samples using Box-Muller
        Z1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
        Z2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)

        # Add samples to result, and remove last one if size
        # requested was odd
        result[: Nsamples // 2] = Z1
        result[Nsamples // 2 :] = Z2[: Nsamples // 2 - odd]

        # Scale results to mean and std
        return mean + result * std


def fisher_yates(arr: np.ndarray, inplace: bool = False) -> np.ndarray:
    """
    Shuffle an array using Fisher-Yates shuffling

    Parameters
    ----------
    arr : ndarray
        Array to shuffle
    infplace : bool, optional
        Shuffle in-place, or return a shuffled copy.
        The default is False

    Returns
    -------
    shuffled : ndarray
        if inplace=False:
            Shuffled array of same shape and dtype of `arr`
        if inplace=True:
            None
    """

    if inplace:
        shuffled = arr
    else:
        shuffled = np.copy(arr)

    N = len(shuffled)
    generator = Random()
    for i in range(N - 1, 0, -1):
        # Get random index
        j = generator.randint(0, i)

        # Swap places
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

    # Only return if not modifying original array
    if not inplace:
        return shuffled


def choice(arr: np.ndarray, size: int = 1) -> np.ndarray:
    """
    Shuffle an array using Fisher-Yates shuffling

    Parameters
    ----------
    arr : ndarray
        Array to shuffle
    size : int, optional
        Number of elements to pick from array
        The default is 1

    Returns
    -------
    shuffled : ndarray
        Shuffled array with size elements
    """
    if size >= len(arr):
        raise ValueError("Cannot request more samples than array is long.")
    shuffled = fisher_yates(arr)
    return shuffled[:size]
