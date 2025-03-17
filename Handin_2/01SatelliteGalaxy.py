import numpy as np


def n(x, A, Nsat, a, b, c):
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


def main():
    from helperscripts.integrate import romberg
    from helperscripts.sampling import rejection
    import matplotlib.pyplot as plt

    # Default values given in problemset
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    bounds = (0, 5)

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
    print(f"∫∫∫n(x) dV = ⟨Nsat⟩ : {np.isclose(Nsat, integrated_Nsat)}")
    
    #########
    ## Q1b ##
    #########
    # Number of galaxies within radius x. Equivalent to integrand, but with correct norm
    pmax = lambda A, a, b, c: 4 * np.pi * A * b**2 * ((a-1)/c)**((a-1)/c) * np.exp((1-a)/c)
    p_of_x = lambda x, *args: 4 * np.pi * x**2 * n(x, A, 1, *args)
    
    
    Nsamples = 10_000
    xx = np.linspace(1e-4, 5, Nsamples)

    # Numerically determine maximum
    pmax = np.max(p_of_x(xx, a, b, c))

    p_of_x_norm = lambda x, *args: p_of_x(x, *args) / pmax
    random_samples = rejection(p_of_x_norm, 1e-4, 5, Nsamples, args=(a, b, c))
    
    xx = np.linspace(1e-4, 5, Nsamples)

    edges = np.logspace(np.log10(1e-4), np.log10(5), 21)
    hist = np.histogram(random_samples, bins=edges)[0]
    hist_scaled = hist / np.diff(edges) / Nsamples

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.stairs(hist_scaled, edges=edges, fill=True, label="Satellite galaxies")
    ax.plot(xx, p_of_x(xx, a, b, c))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 10)
    plt.show()

if __name__ in ("__main__"):
    main()
