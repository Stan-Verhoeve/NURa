from .random import Random
from numpy import zeros, uint64, ndarray


def rejection(dist: callable, low: float, high: float, Nsamples: int, seed: uint64=None, args: tuple=()) -> ndarray:
    """
    Sample a distribution using rejection sampling

    Parameters
    dist : callable
        Distribution to sample
    low : float
        Low end of samples
    high : float
        High end of samples
    Nsamples : int
        Number of samples
    seed : uint64, optional
        Seed for random generators.
        Should only be used when reproducability is desired.
        The default is None
    args : tuple, optional
        Arguments to be passed to dist

    Returns
    -------
    accepted : ndarray
        Accepted values of size Nsamples
    """
    # Provide seed to uniform if given
    if seed:
        U1 = Random(seed)
        # Make sure they do not have the same seed
        U2 = Random(seed >> 8)
    else:
        U1 = Random()
        U2 = Random()
    
    # Pre-allocate accepted sample array
    accepted = zeros(Nsamples)
    n_accepted = 0

    while n_accepted < Nsamples:
        x = U1.uniform(low, high, Nsamples)
        y = U2.uniform(size=Nsamples)
        
        # Mask where we should accept
        to_accept = y < dist(x, *args)
        new_x = x[to_accept]
        
        # Remaining free spaces in sampled array
        remaining = Nsamples - n_accepted
        to_add = min(len(new_x), remaining)
        
        # Fill remaining spaces with accepted samples
        accepted[n_accepted : n_accepted + to_add] = new_x[:to_add]
        n_accepted += to_add

    return accepted
