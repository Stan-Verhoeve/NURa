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
    from helperscripts.optimize import golden_section, levenberg_marquardt
    from helperscripts.io import readfile
    import matplotlib.pyplot as plt


    ##########################
    ## Q1a: finding maximum ##
    ##########################
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
    func = lambda x, *args: -(x**2) * n(x, 1, 1, *args)
    N_of_x = lambda x, *args: 4 * np.pi * x**2 * n(x, A, Nsat, *args)

    # Find minimum of func (maximum of N(x))
    xmin = golden_section(func, *bracket, args=(a, b, c), atol=1e-8)
    print(f"Maximum found at x={xmin}")
    print(f"Function value at maximum: N(x) = {N_of_x(xmin, a, b, c)}")
    
    
    ###########################
    ## Q1b: Gaussian fitting ##
    ###########################
    
    # Create figure
    fig, axs = plt.subplots(3,2, figsize=(1.5*6.4, 1.5*8.0))
    
    for i in range(5):
        print(f"Currently working on satgals_m1{i+1}.txt")
        # Reading data
        radius, nhalo = readfile(f"data/satgals_m1{i+1}.txt")
        
        # Binning the data
        Nbins = 100
        edges = np.logspace(np.log10(1e-4), np.log10(5), Nbins + 1)
        centers_log = 0.5 * (np.log10(edges[1:]) + np.log10(edges[:-1]))
        centers = 10 ** centers_log
        
        # Normalised histogram
        hist = np.histogram(radius, bins=edges)[0]
        hist_scaled = hist / np.diff(edges) / nhalo
        
        # Averagey galaxies per halo
        Ntest = len(radius) / nhalo

        
        #######################
        ## Levenberg fitting ##
        #######################
        
        # TODO: Move these functions to helperscripts?

        # Gradient wrt model parameters for Gaussian errors
        def gauss_grad(data, model, sigma, derivatives, p):
            x, y = data[:,0], data[:,1]
            f = model(x, *p)

            # Jacobian
            J = [df(x, *p) for df in derivatives]
            J = np.stack(J, axis=1)
            res = (y - f) / sigma**2

            return -2 * J.T @ res
        
        # Gradient wrt model parameters for Poissonian errors
        def poiss_grad(data, model, sigma, derivatives, p):
            x, y = data[:,0], data[:,1]
            f = model(x, *p)
            print(f)
            print(p)
            # Jacobian
            J = [df(x, *p) for df in derivatives]
            J = np.stack(J, axis=1)
            res = (y / f - 1)
            return J.T @ res
        
        # logL for Gaussian errors (chiSquared)
        def logL(data, model, sigma, p):
            x, y = data[:,0], data[:,1]
            res = (y - model(x, *p)) / sigma
            return np.sum(res**2)
        
        # logL for Poissonian errors
        def logL_poisson(data, model, sigma, p):
            x, y = data[:, 0], data[:, 1]
            y_model = model(x, *p)
            return np.sum(y * np.log(y_model + 1e-10) - y_model)
        
        
        ##########################################
        ## Model and derivatives wrt parameters ##
        ##########################################
        # TODO: See if possible to move to helperscript?

        def model(x, a, b, c):
            return 4*np.pi * x**2 * n(x, 1, Ntest, a, b, c)
        
        def dn_da(x, a, b, c):
            return model(x, a, b, c) * np.log(x/b)

        def dn_db(x, a, b, c):
            return model(x, a, b, c) * (c * (x/b)**c - (a-3)) / b
        
        def dn_dc(x, a, b, c):
            return -model(x, a, b, c) * np.log(x/b) * (x/b)**c

        
        # Initial guess and data matrix
        p0 = [1.5, 0.5, 1.5]
        data = np.stack([centers, hist_scaled], axis=1)
        
        # Levenberg-Marquardt fitting procedure
        params = levenberg_marquardt(data=data,
                                     model=model,
                                     sigma=Ntest,
                                     derivatives=(dn_da, dn_db, dn_dc),
                                     logL=logL,
                                     dlogL_dp=gauss_grad,
                                     p0=p0,
                                     step=1e-2,
                                     weight=20,
                                     max_iters=300,
                                     atol=0.01)
        
        # TODO: Currently compares to curve_fit
        #       Keep in as comparison? Or remove later?
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(model, data[:,0], data[:,1], p0)
        
        print("    Best fitting parameters using Levenberg-Marquardt")
        print(f"        a={params[0]}")
        print(f"        b={params[1]}")
        print(f"        c={params[2]}")
        print("\n    Best fitting parameters using scipy.optimize.curve_fit")
        print(f"        a={popt[0]}")
        print(f"        b={popt[1]}")
        print(f"        c={popt[2]}")

        row = i//2
        col=i%2
        axs[row,col].set(title=fr"$M_h \approx 10^{{{11+i}}} M_{{\odot}}/h$",
                        xlabel="x",
                        ylabel=r"N/$\langle N_\text{sat}\rangle$",
                        xscale="log",
                        yscale="log",
                        xlim=(1e-4, 5),
                        ylim=(1e-3, 2*max(hist_scaled))
                       )

        axs[row,col].stairs(hist_scaled, edges=edges)
        axs[row,col].plot(xx, model(xx, *params), c="r", label="Levenberg-Marquardt")
        axs[row,col].plot(xx, model(xx, *popt), c="k", ls="--", label="scipy curve_fit")
    
    axs[0,0].legend()
    axs[2,1].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"figures/subplots_fitted", bbox_inches="tight", dpi=600)

if __name__ in ("__main__"):
    main()
