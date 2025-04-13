def main():
    import numpy as np

    from helperscripts.io import readfile
    from helperscripts.prettyprint import pretty_print_title
    
    from helperscripts.optimize import levenberg_marquardt
    from helperscripts.satellite import dmodel_dparams_binned, binned_model
    from helperscripts.likelihoods import (
        gaussian_logL,
        gaussian_logL_gradient,
    )
    from helperscripts.stat import G_test
    
    from helperscripts.integrate import romberg
    
    import matplotlib.pyplot as plt
    import time
    

    ###########################
    ## Q1b: Gaussian fitting ##
    ###########################

    total_time_start = time.time()
    # Create figure
    fig, axs = plt.subplots(3, 2, figsize=(1.5 * 6.4, 1.5 * 8.0))

    # Iterate over the files
    for i in range(5):
        pretty_print_title(f"Currently working on satgals_m1{i+1}.txt")
        
        ##########
        ## Data ##
        ##########

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
        ## Model preparation ##
        #######################

        # Grab the binned model derivatives...
        dbinned_da = lambda x, a, b, c: nhalo * dmodel_dparams_binned(x, Nsat, a, b, c, "a")
        dbinned_db = lambda x, a, b, c: nhalo * dmodel_dparams_binned(x, Nsat, a, b, c, "b")
        dbinned_dc = lambda x, a, b, c: nhalo * dmodel_dparams_binned(x, Nsat, a, b, c, "c")
        # ... and save in tuple for easy passing
        derivatives = [dbinned_da, dbinned_db, dbinned_dc]

        # The model to-be-fitted should have Nsat fixed
        fit_model = lambda edges, a, b, c: nhalo * binned_model(edges, Nsat, a, b, c)
        
        # Expected standard deviation
        def sigma(x, a, b, c):
            return np.sqrt(fit_model(x, a, b, c))
        

        #######################
        ## Levenberg fitting ##
        #######################
        
        # Initial guess and data matrix
        data = [edges, hist]
        p0 = [2, 1, 3]
        
        # Time for information
        fitting_time_start = time.time()
        # Optimal model parameters
        params = levenberg_marquardt(
            data=data,
            model=fit_model,
            sigma=sigma,
            derivatives=derivatives,
            logL=gaussian_logL,
            dlogL_dp=gaussian_logL_gradient,
            p0=p0,
            step=1e-3,
            weight=10,
            DoF=Nbins - 4,
            max_iters=20,
            atol=0.01,
        )

        # Time for information
        fitting_time = time.time() - fitting_time_start
        print(f"    Fitting took {fitting_time:.3f} seconds")
        
        ################
        ## Statistics ##
        ################
        chi_squared = gaussian_logL(data, fit_model, sigma(edges, *params), params)
        chi_squared_reduced = chi_squared / (Nbins - 4)  # 4 degrees of freedom
        
        print("\n    Chi-square        : ", chi_squared)
        print("    Reduced chi-square: ", chi_squared_reduced)
        
        # For G-test, model and fit should sum to the same value
        Gmodel_norm = fit_model(edges, *params)
        Gmodel_norm /= (np.sum(Gmodel_norm) / np.sum(hist))

        G_value, p_value = G_test(hist, Gmodel_norm, DoF=Nbins-4)
        print("\n    G-value: ", G_value)
        print("    p-value: ", p_value)
        
        # Show best-fitting parameters
        print("\n    Best fitting parameters using Levenberg-Marquardt")
        print(f"        a={params[0]}")
        print(f"        b={params[1]}")
        print(f"        c={params[2]}")
        

        # Plot the result
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
            hist / np.diff(edges) / nhalo,
            edges=edges,
            ec="k",
            label="Binned data",
        )
        axs[row, col].stairs(
            fit_model(edges, *params) / np.diff(edges) / nhalo,
            edges=edges,
            ec="r",
            label="Best-fit profile (Levenberg-Marquardt)",
        )

    # Legend doesn't always show for some reason...
    # handles, labels = axs[2, 0].get_legend_handles_labels()
    # axs[2,1].legend(handles, labels, loc=(0.4, 0.15))

    # ... so use this instead
    fig.tight_layout()
    axs[2, 0].legend(loc="center left", bbox_to_anchor=(1.2, 0.5))
    axs[2, 1].set_visible(False)

    fig.savefig(f"figures/02_satellite_galaxies_gaussian_fit", bbox_inches="tight", dpi=600)
    stopTime = time.time()

    total_time = time.time() - total_time_start
    print(f"That took {total_time:.3f} seconds, or {total_time / 60:.3f} minutes")

if __name__ in ("__main__"):
    main()
