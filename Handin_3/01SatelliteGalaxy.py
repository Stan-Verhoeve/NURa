def main():
    import numpy as np
    from helperscripts.integrate import romberg
    from helperscripts.optimize import golden_section, levenberg_marquardt
    from helperscripts.likelihoods import gaussian_logL, gaussian_logL_gradient, poissonian_logL, poissonian_logL_gradient
    from helperscripts.io import readfile
    import matplotlib.pyplot as plt
    import time
    
    # Order of Romberg integration
    ORDER = 4


    ##########################
    ## Q1a: finding maximum ##
    ##########################
    # Default values given in problemset
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    A = 256 / (5 * np.pi**1.5)
    
    # Function to minimize. This is -x^2 n(x)
    # Move 4pi ou, and reintroduce it in the end result only
    func = lambda x, *args: -(x**2) * n(x, 1, 1, *args)
    N_of_x = lambda x, *args: 4 * np.pi * x**2 * n(x, A, Nsat, *args)
    

    # TODO: double-check bracket?
    #       use other minimization routine?
    #       Seems to be fine for now
    bracket = (0.1, 0.2)

    # Find minimum of func (maximum of N(x))
    xmin = golden_section(func, *bracket, args=(a, b, c), atol=1e-8)
    print(f"Maximum found at x={xmin}")
    print(f"Function value at maximum: N(x) = {N_of_x(xmin, a, b, c)}")

    
    ###########################
    ## Q1b: Gaussian fitting ##
    ###########################

    startTime = time.time()
    # Create figure
    fig, axs = plt.subplots(3, 2, figsize=(1.5 * 6.4, 1.5 * 8.0))

    for i in range(5):
        print(f"Currently working on satgals_m1{i+1}.txt")
        # Reading data
        radius, nhalo = readfile(f"data/satgals_m1{i+1}.txt")

        # Binning the data
        Nbins = 100
        edges = np.logspace(np.log10(1e-4), np.log10(5), Nbins + 1)
        centers_log = 0.5 * (np.log10(edges[1:]) + np.log10(edges[:-1]))
        centers = 10**centers_log

        # Histogram
        hist = np.histogram(radius, bins=edges)[0]

        # Averagey galaxies per halo
        Nsat = len(radius) / nhalo

        #######################
        ## Levenberg fitting ##
        #######################

        ##########################################
        ## Model and derivatives wrt parameters ##
        ##########################################
        # TODO: See if possible to move to helperscript?
        # TODO: Highly, highly inefficient, due to (re)calculating
        #       Z and dZ three times (once for each fit param), even
        #       though Z and dZ do not change per fit iteration. So
        #       large improvement to be made here
        def galaxy_dist(x, Nsat, a, b, c):
            """Non-normalised galaxy dist"""
            return 4 * np.pi * Nsat * x ** (a-1) * b ** (3-a) * np.exp(-((x/b)**c))

        def dgalaxy_dparam(x, Nsat, a, b, c, which="a"):
            """Derivative of non-normalised dist wrt its params"""
            if which == "a":
                extra_term = np.log(x/b)
            if which == "b":
                extra_term = (c * (x/b)**c - (a-3)) / b
            if which == "c":
                extra_term = -1 * np.log(x/b) * (x/b)**c

            return galaxy_dist(x, Nsat, a, b, c) * extra_term

        def partition(a, b, c):
            """Normalisation partition"""
            integrand = lambda x: galaxy_dist(x, 1, a, b, c)

            return romberg(integrand, (1e-4, 5), m=ORDER)

        def dpartition_dparams(a, b, c):
            """Partition derivative wrt one of its params"""
            df_da = lambda x: dgalaxy_dparam(x, 1, a, b, c, which="a")
            df_db = lambda x: dgalaxy_dparam(x, 1, a, b, c, which="b")
            df_dc = lambda x: dgalaxy_dparam(x, 1, a, b, c, which="c")

            dpart_da = romberg(df_da, (1e-4, 5), m=ORDER)
            dpart_db = romberg(df_db, (1e-4, 5), m=ORDER)
            dpart_dc = romberg(df_dc, (1e-4, 5), m=ORDER)

            return [dpart_da, dpart_db, dpart_dc]

        def model(x, Nsat, a, b, c):
            """Normalised model"""
            return galaxy_dist(x, Nsat, a, b, c) / partition(a, b, c)

        def dmodel_dparam(x, Nsat, a, b, c, which="a"):
            """Model derivative wrt one of its params"""
            dZ = dpartition_dparams(a, b, c)
            Z = partition(a, b, c)
            if which == "a":
                extra_term = np.log(x/b)
                dZ = dZ[0]
            if which == "b":
                extra_term = (c * (x/b)**c - (a-3)) / b
                dZ = dZ[1]
            if which == "c":
                extra_term = -1 * np.log(x/b) * (x/b)**c
                dZ = dZ[2]

            # Product rule
            return galaxy_dist(x, Nsat, a, b, c) * (extra_term/Z - dZ/Z**2)

        def bin_function(func, binedges):
            """Bin function given binedges"""
            N = len(binedges) - 1
            result = np.zeros(N)

            for i in range(N):
                result[i] = romberg(func, (binedges[i], binedges[i+1]), m=ORDER)

            return result * nhalo

        def binned_model(binedges, Nsat, a, b, c):
            """Binned galaxy model"""
            func = lambda x: model(x, Nsat, a, b, c)
            return bin_function(func, binedges)

        def dmodel_dparams_binned(binedges, Nsat, a, b, c, which):
            """Binned derivative wrapper"""
            dm_dp = lambda x: dmodel_dparam(x, Nsat, a, b, c, which=which)

            return bin_function(dm_dp, binedges)

        # Grab the binned model derivatives...
        dbinned_da = lambda x, a, b, c: dmodel_dparams_binned(x, Nsat, a, b, c, "a")
        dbinned_db = lambda x, a, b, c: dmodel_dparams_binned(x, Nsat, a, b, c, "b")
        dbinned_dc = lambda x, a, b, c: dmodel_dparams_binned(x, Nsat, a, b, c, "c")
        # ... and save in tuple for easy passing
        derivatives = [dbinned_da, dbinned_db, dbinned_dc]

        # The model to-be-fitted should have Nsat fixed
        fit_model = lambda edges, a, b, c: binned_model(edges, Nsat, a, b, c)
    
        # Expected standard deviation
        def sigma(x, a, b, c):
            return np.sqrt(fit_model(x, a, b, c))
        
        # TODO: find a way to make it work with this theory?
        #       Current problem: makes it so that fitting
        #       procedure assumes a theory model, instead of data etc
        # from helperscripts.satellite import GalaxyDistribution
        # theory = GalaxyDistribution(ORDER)
        # theory.theta = (a, b, c)
        
        # Initial guess and data matrix
        data = [edges, hist]
        p0 = [2, 1, 3]
        
        # TODO: Come back to this
        # from MCMC import metropolis_hastings_fit
        # params = metropolis_hastings_fit(data, binned_model, gaussian_logL, 0.1, p0, num_iterations=100_000, num_chains=1, step_size=0.05)
        
        # Levenberg-Marquardt fitting procedure
        # Model parameters
        params = levenberg_marquardt(
            data=data,
            model=fit_model,
            sigma=sigma,  # np.sqrt(Ntest),
            derivatives=derivatives,
            logL=gaussian_logL,
            dlogL_dp=gaussian_logL_gradient,
            p0=p0,
            step=1e-3,
            weight=10,
            DoF=Nbins-4,  # We have 3 params, so intuitively Nbins - 3. However, once Nbins-1 have been filled, the last one is fixd
            max_iters=10,
            atol=0.01,
        )
        
        from helperscripts.TEMP import lucas
        print("chi2 own  ", gaussian_logL(data, fit_model, sigma(edges, *params), params))
        print("chi2 lucas", gaussian_logL(data, fit_model, sigma(edges, *lucas[i]), lucas[i]))
        
        # TODO: Currently compares to curve_fit
        #       Keep in as comparison? Or remove later?
        from scipy.optimize import curve_fit

        popt, pcov = curve_fit(
            fit_model, edges, hist, p0, sigma=np.sqrt(Nsat)
        )

        print("    Best fitting parameters using Levenberg-Marquardt")
        print(f"        a={params[0]}")
        print(f"        b={params[1]}")
        print(f"        c={params[2]}")
        print("\n    Best fitting parameters using scipy.optimize.curve_fit")
        print(f"        a={popt[0]}")
        print(f"        b={popt[1]}")
        print(f"        c={popt[2]}")

        row = i // 2
        col = i % 2
        axs[row, col].set(
            title=f"$M_h \\approx 10^{{{11+i}}} M_{{\\odot}}/h$",
            xlabel="x",
            ylabel=r"N",
            xscale="log",
            yscale="log",
            xlim=(1e-4, 5),
            ylim=(1e-3, 2 * max(hist / np.diff(edges) / nhalo)),
        )

        axs[row, col].stairs(
            hist / np.diff(edges) / nhalo, edges=edges, label="Binned data",
        )
        # axs[row, col].stairs(
        #     binned_model(centers, *popt) / np.diff(edges) / nhalo,
        #     edges=edges,
        #     lw=5,
        #     ec="gray",
        #     alpha=0.5,
        #     label="Best-fit profile (scipy curve_fit)",
        # )
        axs[row, col].stairs(
            fit_model(edges, *params) / np.diff(edges) / nhalo,
            edges=edges,
            ec="k",
            label="Best-fit profile (Levenberg-Marquardt)",
        )
        axs[row, col].stairs(
            fit_model(edges, *lucas[i]) / np.diff(edges) / nhalo,
            edges=edges,
            ec="r",
            label="Lucas",
        )
    handles, labels = axs[2, 0].get_legend_handles_labels()
    fig.tight_layout()
    axs[2,0].legend(loc="center left", bbox_to_anchor=(1.2, 0.5))
    # axs[2,1].legend(handles, labels, loc=(0.4, 0.15))
    # plt.figlegend(handles, labels, loc=(0.4, 0.15))
    # plt.figlegend(handles, labels, loc='lower left')#, bbox_to_anchor=(0.4, 0.15))
    axs[2, 1].set_visible(False)

    fig.savefig(f"figures/subplots_fitted", bbox_inches="tight", dpi=600)
    stopTime = time.time()

    totalTime = stopTime - startTime
    print(f"That took {totalTime} seconds, or {totalTime / 60} minutes")


if __name__ in ("__main__"):
    main()
