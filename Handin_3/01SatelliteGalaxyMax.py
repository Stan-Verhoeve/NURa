def main():
    import numpy as np
    from helperscripts.satellite import n
    from helperscripts.optimize import golden_section
    import matplotlib.pyplot as plt

    # Default values given in problemset
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    A = 256 / (5 * np.pi**1.5)

    # Function to minimize. This is -N(x) = -x^2 n(x) (maximizes N(x))
    # Move 4pi out, and reintroduce it in the end result only
    func = lambda x, *args: -(x**2) * n(x, 1, 1, *args)
    N_of_x = lambda x, *args: 4 * np.pi * x**2 * n(x, A, Nsat, *args)

    # Initial bracket. Maximum lies between these
    bracket = (0.1, 0.2)

    # Find minimum of func (maximum of N(x))
    xmin = golden_section(func, *bracket, args=(a, b, c), atol=1e-8)

    # Print results
    print(f"Maximum found at x={xmin}")
    print(f"Function value at maximum: N(x) = {N_of_x(xmin, a, b, c)}")

if __name__ in ("__main__"):
    main()
