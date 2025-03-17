import numpy as np


class MWC:
    def __init__(self, seed, a):
        if not 0 < seed < 2**32:
            raise ValueError("Seed should be 0 < seed < 2**32.")

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
        self.state = self.a * (self.state & np.uint64(2**32 - 1)) + (self.state >> np.uint64(32))
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
    def __init__(self, seed):
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
        self.lcg_state = xorshift(sub1_state, 13, 17, 5)  # Values for a taken from wiki

        # Result of second sub-generator
        self.xor_state = xorshift(self.xor_state, 13, 17, 5) # Values for a taken from wiki

        # Result of third sub-generator
        mwc_state = self.mwc.random()
        
        # Final output
        out = (self.lcg_state + self.xor_state) ^ mwc_state

        return out & np.uint64(2**32-1)
    
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
