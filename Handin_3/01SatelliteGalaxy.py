import numpy as np

def n(
    x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> np.ndarray:
    """
    Number density profile of satellite galaxies

    Parameters
    ----------
    x : float | ndarray
        Radius in units of virial radius; x = r / r_virial
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    float | ndarray
        Same type and shape as x. Number density of satellite galaxies
        at given radius x.
    """
    return A * Nsat * ((x / b) ** (a - 3)) * np.exp(-((x / b) ** c))


def dn_dx(x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float):
    """
    Derivative of number density provide

    Parameters
    ----------
    x : float | ndarray
        Radius in units of virial radius; x = r / r_virial
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    float | ndarray
        Same type and shape as x. Derivative of number density of
        satellite galaxies at given radius x.
    """
    return (
        -A
        * Nsat
        * b**3
        * (x / b) ** (a)
        * (c * (x / b) ** c - a + 3)
        * np.exp(-((x / b) ** c))
        / x**4
    )


def main():
    from helperscripts.integrate import romberg
    from helperscripts.optimize import golden_section

    # Default values given in problemset
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    A = 256 / (5 * np.pi**1.5)

    # TODO: double-check bracket?
    #       use other minimization routine?
    bracket = (0.1, 0.2)

    xx = np.linspace(1e-8, 5, 1000)

    # Function to minimize. This is -x^2 n(x)
    # Move 4pi ou, and reintroduce it in the end result only
    func = lambda x, *args: -x**2 * n(x, 1, 1, *args)
    N_of_x = lambda x, *args: 4*np.pi*x**2 * n(x, A, Nsat, *args)
    
    # Find minimum of func (maximum of N(x))
    xmin = golden_section(func, *bracket, args=(a,b,c), atol=1e-8)
    print(f"Maximum found at x={xmin}")
    print(f"Function value at maximum: N(x) = {N_of_x(xmin, a, b, c)}")

    
    # TODO: test. Remove before handing in
    from matplotlib import pyplot as plt
    
    plt.figure()
    plt.plot(xx, func(xx, a, b, c))
    plt.scatter(xmin, func(xmin, a, b, c), c="r")
    plt.xscale("log")
    # plt.yscale("log")
    plt.savefig("figures/tests/01Satellite_test.png")


if __name__ in ("__main__"):
    main()
