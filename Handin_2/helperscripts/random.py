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


class MWC:
    def __init__(self, seed, a):
        if not 0 < seed < 2**32:
            seed = np.uint64(seed) & (2**32 - 1)

        self.state = np.uint64(seed)
        self.a = np.uint64(a)

    def next(self):
        """
        Get next pseudo-random number in sequence

        Returns
        -------
        self.state
            Current state of the generator
        """
        self.state = self.a * (self.state & np.uint64(2**32 - 1)) + (
            self.state >> np.uint64(32)
        )
        return self.state

    def random(self):
        return self.next() & np.uint64(2**32 - 1)


def xorshift(seed, a1, a2, a3):
    state = np.uint64(seed)
    a1 = np.uint64(a1)
    a2 = np.uint64(a2)
    a3 = np.uint64(a3)

    state = state ^ (state >> a1)
    state = state ^ (state << a2)
    state = state ^ (state >> a3)

    return state


def LCG(seed, a, c, m):
    state = np.uint32(seed)
    a = np.uint64(a)
    c = np.uint64(c)
    m = np.uint64(m)

    state = np.uint64(a * state + c) % m

    return state


class Random:
    def __init__(self, seed=None):
        if not seed:
            seed = get_time_based_seed()
        self.lcg_state = np.uint64(seed)
        self.xor_state = np.uint64(seed + 1)
        self.mwc = MWC(seed + 2, 4294957665)

    def next(self):
        """
        Get next pseudo-random number in sequence

        Returns
        -------
        out
            Current state of the generator
        """
        # Result of first sub-generator
        sub1_state = LCG(self.lcg_state, 1_664_525, 1_013_904_223, 2**32)
        # Feed result of LCG into XOR-shift
        self.lcg_state = xorshift(
            sub1_state, 13, 17, 5
        )  # Values for a taken from wiki

        # Result of second sub-generator
        self.xor_state = xorshift(
            self.xor_state, 13, 17, 5
        )  # Values for a taken from wiki

        # Result of third sub-generator
        mwc_state = self.mwc.random()

        # Final output
        out = (self.lcg_state & self.xor_state) ^ mwc_state

        return out & np.uint64(2**32 - 1)

    def random(self):
        # Divide by 2 ** 32 to get in [0, 1)
        return self.next() / 2**32

    def uniform(self, low=0, high=1, size=1):
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
        arr = np.array([self.random() for _ in range(size)])

        # Scale default [0, 1) to [low, high)
        arr = arr * (high - low) + low

        return arr

    def randint(self, low=0, high=10, size=1):
        return np.int32(self.uniform(low, high))


def fisher_yates(arr, inplace=False):
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
        Shuffled array of same shape and dtype of `arr`
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


def choice(arr, size=1):
    """
    Grab random samples from array
    """
    if size >= len(arr):
        raise ValueError("Cannot request more samples than array is long.")
    shuffled = fisher_yates(arr)
    return shuffled[:size]
