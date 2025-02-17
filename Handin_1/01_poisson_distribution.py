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

    # num = l**k * exp(-l)
    # denom = k!

    # Convert to log_2 space:
    # log2(x) = ln(x) / ln(2)
    # TODO: can do with np.log() for ln
    #       but check if log2() is faster
    #       and if we can do bitshifting magic
    #       for now use np.log()
    
    # log(a * b) == log(a) + log(b)
    # log(l**k) == k * log(l)
    # log(exp(-l)) == -l
    # Therefore log(l**k * exp(-l)) == k * log(l) - l

    # Explicity convert k and lbd to 32bit
    k = int32(k)
    lbd = float32(lbd)
    
    # log of numerator
    log_num = float32(k * log(lbd) - lbd)

    # k! == k (k-1) (k-2) ... (2)
    # In logspace, becomes a sum:
    # log(k!) = log(k) + log(k-1) + log(k-2) ...
    # TODO: check if np.sum is allowed

    # Create arange from 2 to k
    ks = arange(2, k+1, dtype=int32)
    
    # log of denominator
    # Take log and sum to get factorial
    log_denom = int32(log(ks).sum())

    # Take difference of num and denom, since log(num/denom) == log(num) - log(denom)
    logP = float32(log_num - log_denom)

    return logP

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
