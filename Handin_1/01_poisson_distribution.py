from numpy import log, arange, int32, float32


def poisson(k: int32, lbd: float32) -> float32:
    """
    Poisson distribution for k and lambda

    Parameters
    ----------
    k: int32
        Mean number of events
    lbd: float32
        lambda of the distribution

    Returns
    -------
    logP: float32
        Natural log of P
    """
    if k < 0:
        raise ValueError("k should be non-negative")
    # TODO: Make vectorised?

    # Explicity convert k and lbd to 32bit
    k = int32(k)
    lbd = float32(lbd)

    # log of numerator
    log_num = float32(k * float32(log(lbd)) - lbd)

    # With an eye to memory efficiency, as opposed to runtime speed,
    # use a for-loop to get the log of denominator
    log_denom = float32(0.0)

    for ks in range(2, k + 1):
        log_denom += float32(log(ks))

    # Take difference of num and denom, since log(num/denom) == log(num) - log(denom)
    logP = log_num - log_denom

    return float32(logP)


def main():
    from numpy import exp

    # Values in problemset
    # TODO: No longer hard-code these. Instead, make arange / linspace?
    values = [(1, 0), (5, 10), (3, 21), (2.6, 40), (100, 5), (101, 200)]

    # Iterate over lambda- and k-values
    for lbd, k in values:
        logP = poisson(k, lbd)
        print(f"({lbd}, {k}): P = {exp(logP):.6e}")


if __name__ in ("__main__"):
    main()
