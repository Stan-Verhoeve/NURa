import numpy as np

def pearson(x : np.ndarray, y : np.ndarray = None):
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
        raise ValueError(f"Shape of `x` and `y` should be the same, but got {np.shape(x)} and {np.shape(y)}")
    xy_mean = np.mean(x * y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.var(x)
    y_var = np.var(y)

    denom_inv = np.sqrt(x_var * y_var) ** (-1)

    r_xy = (xy_mean - x_mean * y_mean) * denom_inv

    return r_xy

class LCG:
    """
    Linear congruential generator
    """
    def __init__(self, seed, a, c, m):
        self.a = a
        self.c = c
        self.m = m

        # Current state of generator
        self.state = seed
    def next(self):
        """
        Get next pseudo-random number in sequence

        Returns
        -------
        self.state
            Current state of the generator
        """
        self.state = (self.a * self.state + self.c) % self.m
        return self.state
    
    def random(self):
        return self.next() / self.m

    def generate(self, size : int) -> np.ndarray:
        """
        Generate array of random numbers

        Parameters
        ----------
        size : int
            Size of array

        Returns
        -------
        arr : ndarray
            Array containing pseudo-random numbers
        """
        arr = np.array([self.random() for _ in range(size)])
        return arr

def lcg_tester(a, c, m, N=200, seed=1234):
    lcg = LCG(seed, a, c, m)
    X = lcg.generate(size=N)
    x = X[1:]
    y = X[:-1]
    r = pearson(x, y)
    
    # For pretty printing
    space = ""
    if r >= 0:
        space = " "
    print(f"Pearson coefficient: {space}{r:.2e} (LCG with a={a}, c={c}, m={m})")

def main():
    ##########
    ## LCG ###
    ##########
    lcg_tester(a=5, c=10, m=7)
    lcg_tester(a=5, c=10, m=2**32)
    lcg_tester(a=1_664_525, c=1_013_904_223, m=2**32)

    ###################
    ## Numpy random ###
    ###################
    X = np.random.uniform(size=200)
    x = X[1:]
    y = X[:-1]
    r = pearson(x, y)

    # For pretty printing
    space = ""
    if r >= 0:
        space = " "
    print(f"Pearson coefficient: {space}{r:.2e} (numpy.random.uniform)")

if __name__ in ("__main__"):
    main()
