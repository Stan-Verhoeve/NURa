import numpy as np

def romberg(func, bounds, m=5, err=False, args=None):
    # Extract bounds and first step size
    a, b = bounds
    h = b - a
    N = 1
    
    # Array to hold integral guesses
    r = np.zeros(m)

    # Initial trapezoid (on full domain)
    r[0] = 0.5 * h * (func(a, *args) + func(b, *args))

    for i in range(1,m):
        hi = h * 2**(-i)
        Delta = 2 * hi
        N *= 2

        # New points to evaluate
        newx = np.arange(a + hi, b, Delta)
        newy = func(newx, *args)
        
        # New estimate
        r[i] = 0.5 * (r[i-1] + Delta * np.sum(newy))
    
    # Combine estimates iteratively
    for i in range(m):
        for j in range(1,m-i):
            denom_inv = (4 ** (i+1) - 1) ** (-1)
            r[j-1] = ((4 ** (i+1)) * r[j] - r[j-1]) * denom_inv
    
    if err:
        return r[0], abs(r[0] - r[1])
    return r[0]

def n(x, a, b, c):
    return ((x / b) ** (a - 3)) * np.exp(-(x / b) ** c)

def func(x, *args):
    return x ** 2

def main():
    a=2.4
    b=0.25
    c=1.6
    bounds = (1e-4, 5)
    
    result, err = romberg(n, bounds, m=20, err=True, args=(a,b,c))
    A = result ** (-1)
    print(result, err)
    from scipy.integrate import quad

    print(quad(n, *bounds, args=(a,b,c)))

    print(A)
    
if __name__ in ("__main__"):
    main()
