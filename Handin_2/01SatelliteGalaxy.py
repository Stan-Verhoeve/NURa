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

    # Default values given in problemset
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    bounds = (1e-4, 5)

    # 1D integrand to solve for.
    integrand = lambda x, *args: 4 * np.pi * x**2 * n(x, 1, 1, *args)
    result, err = romberg(integrand, bounds, m=10, args=(a, b, c), err=True)

    # Normalisation
    A = 1.0 / result
    print(f"Normalisation constant: A = {A}")

    # Validate that number density profile integrates to Nsat
    integrand = lambda x, *args: 4 * np.pi * x**2 * n(x, *args)
    integrated_Nsat = romberg(integrand, bounds, m=10, args=(A, Nsat, a, b, c))
    print("\nSanity check")
    print(f"∫∫∫n(x) dV = ⟨Nsat⟩ : {np.isclose(Nsat, integrated_Nsat)}")


if __name__ in ("__main__"):
    main()
