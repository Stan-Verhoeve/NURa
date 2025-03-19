from .random import Random
from numpy import zeros


def rejection(dist, low, high, Nsamples, seed=None, args=()):
    if seed:
        U1 = Random(seed)
        U2 = Random(seed >> 8)
    else:
        U1 = Random()
        U2 = Random()

    accepted = zeros(Nsamples)
    n_accepted = 0

    while n_accepted < Nsamples:
        x = U1.uniform(low, high, Nsamples)
        y = U2.uniform(size=Nsamples)

        to_accept = y < dist(x, *args)
        new_x = x[to_accept]

        remaining = Nsamples - n_accepted
        to_add = min(len(new_x), remaining)

        accepted[n_accepted : n_accepted + to_add] = new_x[:to_add]
        n_accepted += to_add

    return accepted
