def main():
    import numpy as np

    from helperscripts.io import readfile

    from helperscripts.optimize import (
        golden_section,
        levenberg_marquardt,
        quasi_newton,
    )
    from helperscripts.satellite import dmodel_dparams_binned, binned_model, model
    from helperscripts.likelihoods import (
        gaussian_logL,
        gaussian_logL_gradient,
        poissonian_logL,
        poissonian_logL_gradient,
    )
    from helperscripts.stat import G_test

    from helperscripts.sampling import rejection

    import matplotlib.pyplot as plt
    import time

    WHICH = 4
    ##########
    ## Data ##
    ##########

    # Reading data
    radius, nhalo = readfile(f"data/satgals_m1{WHICH}.txt")

    # Binning the data
    Nbins = 100
    edges = np.logspace(np.log10(1e-4), np.log10(5), Nbins + 1)
    centers_log = 0.5 * (np.log10(edges[1:]) + np.log10(edges[:-1]))
    centers = 10**centers_log

    # Histogram
    hist = np.histogram(radius, bins=edges)[0]

    # Averagey galaxies per halo
    Nsat = len(radius) / nhalo

    best_parameters_gauss = np.load("data/best_parameters_gauss.npy")[WHICH - 1]
    best_parameters_poiss = np.load("data/best_parameters_poiss.npy")[WHICH - 1]

    # Best model
    best_model_gauss = lambda x: model(x, Nsat, *best_parameters_gauss)
    best_model_poiss = lambda x: model(x, Nsat, *best_parameters_poiss)
    xmax_gauss = golden_section(
        lambda x: -1 * best_model_gauss(x), 0.2, 0.6, rtol=1e-8
    )
    xmax_poiss = golden_section(
        lambda x: -1 * best_model_poiss(x), 0.2, 0.6, rtol=1e-8
    )
    model_max_gauss = best_model_gauss(xmax_gauss)
    model_max_poiss = best_model_poiss(xmax_poiss)

    # Normalised model for rejection sampling
    normalised_model_gauss = lambda x: best_model_gauss(x) / model_max_gauss
    normalised_model_poiss = lambda x: best_model_poiss(x) / model_max_poiss

    # Grab the binned model derivatives...
    dbinned_da = lambda x, a, b, c: nhalo * dmodel_dparams_binned(
        x, Nsat, a, b, c, "a", use_integral=False
    )
    dbinned_db = lambda x, a, b, c: nhalo * dmodel_dparams_binned(
        x, Nsat, a, b, c, "b", use_integral=False
    )
    dbinned_dc = lambda x, a, b, c: nhalo * dmodel_dparams_binned(
        x, Nsat, a, b, c, "c", use_integral=False
    )
    # ... and save in tuple for easy passing
    derivatives = [dbinned_da, dbinned_db, dbinned_dc]

    # The model to-be-fitted should have Nsat fixed
    fit_model = lambda edges, a, b, c: nhalo * binned_model(
        edges, Nsat, a, b, c, use_integral=False
    )

    # Expected standard deviation
    def sigma(x, a, b, c):
        return np.sqrt(fit_model(x, a, b, c))

    # Data vector
    data = [edges, hist]

    # Initialise figure
    gauss_fig = plt.figure()
    poiss_fig = plt.figure()
    gauss_ax = gauss_fig.add_subplot(111)
    poiss_ax = poiss_fig.add_subplot(111)

    # Number of MC iterations
    Niters = 100

    # Keep track of parameters
    mc_parameters_gauss = np.zeros((Niters, 3), dtype="float")
    mc_parameters_poiss = np.zeros((Niters, 3), dtype="float")

    p0 = (2, 1, 3)

    start = time.time()
    for i in range(Niters):
        # Sample from the normalised model
        samples_gauss = rejection(normalised_model_gauss, 1e-4, 5, len(radius))
        samples_poiss = rejection(normalised_model_poiss, 1e-4, 5, len(radius))

        hist_gauss, __ = np.histogram(samples_gauss, bins=edges)
        hist_poiss, __ = np.histogram(samples_poiss, bins=edges)

        params_gauss = levenberg_marquardt(
            data=[edges, hist_gauss],
            model=fit_model,
            sigma=sigma,
            derivatives=derivatives,
            logL=gaussian_logL,
            dlogL_dp=gaussian_logL_gradient,
            p0=p0,
            DoF=Nbins - 4,
            step=1e-3,
            weight=10,
            max_iters=20,
            atol=0.01,
        )

        params_poiss = quasi_newton(
            data=[edges, hist_poiss],
            model=fit_model,
            sigma=sigma,
            derivatives=derivatives,
            logL=poissonian_logL,
            dlogL_dp=poissonian_logL_gradient,
            p0=p0,
            max_iters=20,
            atol=1e-5,
        )

        mc_parameters_gauss[i] = params_gauss
        mc_parameters_poiss[i] = params_poiss

        gauss_ax.stairs(
            fit_model(edges, *params_gauss) / np.diff(edges) / nhalo,
            edges=edges,
            ec="gray",
            alpha=0.1,
        )
        poiss_ax.stairs(
            fit_model(edges, *params_poiss) / np.diff(edges) / nhalo,
            edges=edges,
            ec="gray",
            alpha=0.1,
        )

    mc_time = time.time() - start

    # Get average of parameters
    avg_mc_parameters_gauss = np.mean(mc_parameters_gauss, axis=0)
    avg_mc_parameters_poiss = np.mean(mc_parameters_poiss, axis=0)

    ################
    ## Statistics ##
    ################
    chi_squared = gaussian_logL(
        data,
        fit_model,
        sigma(edges, *avg_mc_parameters_gauss),
        avg_mc_parameters_gauss,
    )
    chi_squared_reduced = chi_squared / (Nbins - 4)  # 4 degrees of freedom
    logL = poissonian_logL(
        data,
        fit_model,
        sigma(edges, *avg_mc_parameters_poiss),
        avg_mc_parameters_poiss,
    )

    print("\nChi-square        : ", chi_squared)
    print("Reduced chi-square: ", chi_squared_reduced)
    print("log-L             : ", logL)

    # For G-test, model and fit should sum to the same value
    Gmodel_norm_gauss = fit_model(edges, *avg_mc_parameters_gauss)
    Gmodel_norm_gauss /= np.sum(Gmodel_norm_gauss) / np.sum(hist)
    Gmodel_norm_poiss = fit_model(edges, *avg_mc_parameters_poiss)
    Gmodel_norm_poiss /= np.sum(Gmodel_norm_poiss) / np.sum(hist)

    G_value_gauss, p_value_gauss = G_test(hist, Gmodel_norm_gauss, DoF=Nbins - 4)
    G_value_poiss, p_value_poiss = G_test(hist, Gmodel_norm_poiss, DoF=Nbins - 4)

    print("\nGaussian fit")
    print("    G-value: ", G_value_gauss)
    print("    p-value: ", p_value_gauss)

    print("\nPoissonian fit")
    print("    G-value: ", G_value_poiss)
    print("    p-value: ", p_value_poiss)

    print("\nbest fitting parameters (gaussian)")
    print(f"    a={avg_mc_parameters_gauss[0]}")
    print(f"    b={avg_mc_parameters_gauss[1]}")
    print(f"    c={avg_mc_parameters_gauss[2]}")
    print("\nbest fitting parameters (poissonian)")
    print(f"    a={avg_mc_parameters_poiss[0]}")
    print(f"    b={avg_mc_parameters_poiss[1]}")
    print(f"    c={avg_mc_parameters_poiss[2]}")

    print(
        f"\nMC procedure took {mc_time:.3f} seconds, or {mc_time / 60:.3f} minutes"
    )

    gauss_ax.stairs(
        hist / np.diff(edges) / nhalo, edges=edges, ec="b", label="Binned data"
    )
    poiss_ax.stairs(
        hist / np.diff(edges) / nhalo, edges=edges, ec="b", label="Binned data"
    )

    gauss_ax.stairs(
        fit_model(edges, *best_parameters_gauss) / np.diff(edges) / nhalo,
        edges=edges,
        ec="r",
        label="Best fit (gaussian)",
    )
    poiss_ax.stairs(
        fit_model(edges, *best_parameters_poiss) / np.diff(edges) / nhalo,
        edges=edges,
        ec="r",
        label="Best fit (poissonian)",
    )

    gauss_ax.stairs(
        fit_model(edges, *avg_mc_parameters_gauss) / np.diff(edges) / nhalo,
        edges=edges,
        ec="k",
        label="MC fit (gaussian)",
    )
    poiss_ax.stairs(
        fit_model(edges, *avg_mc_parameters_poiss) / np.diff(edges) / nhalo,
        edges=edges,
        ec="k",
        label="MC fit (poissonian)",
    )

    gauss_ax.legend(loc="upper left")
    poiss_ax.legend(loc="upper left")
    gauss_ax.set(
        title=f"$M_h \\approx 10^{{{10+WHICH}}} M_{{\\odot}}/h$",
        xlabel="x",
        ylabel=r"N",
        xscale="log",
        yscale="log",
        xlim=(1e-4, 5),
        ylim=(1e-3, 2 * max(hist / np.diff(edges) / nhalo)),
    )
    poiss_ax.set(
        title=f"$M_h \\approx 10^{{{10+WHICH}}} M_{{\\odot}}/h$",
        xlabel="x",
        ylabel=r"N",
        xscale="log",
        yscale="log",
        xlim=(1e-4, 5),
        ylim=(1e-3, 2 * max(hist / np.diff(edges) / nhalo)),
    )

    gauss_fig.savefig(
        "figures/04_satellite_galaxies_gaussian_mc", bbox_inches="tight", dpi=600
    )
    poiss_fig.savefig(
        "figures/04_satellite_galaxies_poissonian_mc", bbox_inches="tight", dpi=600
    )


if __name__ in ("__main__"):
    main()
