from .rng import Random
import numpy as np

def rejection(dist, low, high, Nsamples, args=()):
    U1 = Random(44)
    U2 = Random(33)
    
    accepted = np.zeros(Nsamples)
    n_accepted = 0

    while n_accepted < Nsamples:
        x = U1.uniform(low, high)
        y = U2.uniform()
        if x < 3e-3 and y < 3e-2:
            if y <= dist(x, *args):
                print(x)
        
        if y <= dist(x, *args):
            accepted[n_accepted] = x
            n_accepted += 1

    return accepted

