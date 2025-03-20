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
    from helperscripts.sampling import rejection
    from helperscripts.random import choice
    from helperscripts.differentiate import ridder
    from helperscripts.sorting import merge_sort_in_place
    import matplotlib.pyplot as plt

    # Default values given in problemset
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    bounds = (0, 5)
    xmin, xmax = 1e-4, 5
    seed = 1  # For reproducability
    Nsamples = 10_000
    xx = np.linspace(xmin, xmax, Nsamples)

    # 1D integrand to solve for.
    # Move 4pi out of the integrand, and reintroduce it
    # in the end result only
    integrand = lambda x, *args: x**2 * n(x, 1, 1, *args)
    result, err = romberg(integrand, bounds, m=10, args=(a, b, c), err=True)

    # Normalisation
    A = 1 / (4 * np.pi * result)
    print(f"Normalisation constant: A = {A}")

    # Validate that number density profile integrates to Nsat
    integrand = lambda x, *args: 4 * np.pi * x**2 * n(x, *args)
    integrated_Nsat = romberg(integrand, bounds, m=10, args=(A, Nsat, a, b, c))
    print("\nSanity check")
    print(f"int int int n(x) dV = <Nsat> : {np.isclose(Nsat, integrated_Nsat)}")

    #########
    ## Q1b ##
    #########
    # Number of galaxies within radius x. Equivalent to integrand, but with correct norm
    p_of_x = lambda x, *args: 4 * np.pi * x**2 * n(x, A, 1, *args)

    # Numerically determine maximum
    pmax = np.max(p_of_x(xx, a, b, c))

    p_of_x_norm = lambda x, *args: p_of_x(x, *args) / pmax
    random_samples = rejection(
        p_of_x_norm, 1e-4, 5, Nsamples, seed=seed, args=(a, b, c)
    )

    ##############
    ## Plotting ##
    ##############

    edges = np.logspace(np.log10(1e-4), np.log10(5), 21)
    hist = np.histogram(random_samples, bins=edges)[0]
    hist_scaled = hist / np.diff(edges) / Nsamples

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.stairs(hist_scaled, edges=edges, fill=True, label="Satellite galaxies")
    ax.plot(xx, p_of_x(xx, a, b, c), c="r", ls="-", label="Analytic")

    ax.set(
        xlim=(xmin, xmax),
        ylim=(10 ** (-3), 10),
        yscale="log",
        xscale="log",
        xlabel="Relative radius",
        ylabel="Number of galaxies",
    )
    ax.legend()
    fig.savefig("figures/01_satellite_galaxies_Q1b", bbox_inches="tight", dpi=300)

    #########
    ## Q1c ##
    #########
    chosen = choice(random_samples, 100)
    merge_sort_in_place(chosen)

    # Cumulative plot of the chosen galaxies (1c)
    fig, ax = plt.subplots()
    ax.plot(chosen, np.arange(100))
    ax.set(
        xscale="log",
        xlabel="Relative radius",
        ylabel="Cumulative number of galaxies",
        xlim=(xmin, xmax),
        ylim=(0, 100),
    )
    fig.savefig("figures/01_satellite_galaxies_Q1c.png", dpi=300)

    #########
    ## Q1d ##
    #########
    x_to_eval = 1
    func_to_eval = lambda x: n(x, A, Nsat, a, b, c)
    dn_dx_numeric = ridder(func_to_eval, x_to_eval, h_init=0.1, d=2, eps=1e-15)[0]
    dn_dx_analytic = dn_dx(x_to_eval, A, Nsat, a, b, c)

    print("\nDerivative at x=1")
    print(f"    Analytic: {dn_dx_analytic}")
    print(f"    Numeric : {dn_dx_numeric}")
    print(f"    |numeric - analytic| = {abs(dn_dx_analytic - dn_dx_numeric):.3e}")


if __name__ in ("__main__"):
    main()
