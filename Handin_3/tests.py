import numpy as np
import matplotlib.pyplot as plt


def test_bracketing():
    """
    Quick test on finding a bracket for a function
    """
    from helperscripts.optimize import find_bracket

    # Quadratic function with minimum -3 at x=0
    func = lambda x: -3 + np.array(x) ** 2
    xx = np.linspace(-5, 5, 1000)

    initial_bracket = (-2, 3)
    bracket = find_bracket(func, *initial_bracket)
    f_init = func(initial_bracket)
    f_bracket = func(bracket)
    print("Initial bracket")
    print(f"    x = {initial_bracket}")
    print(f"    y = {f_init}")
    print("Final bracket")
    print(f"    x = {bracket}")
    print(f"    y = {f_bracket}")

    plt.figure()
    plt.plot(xx, func(xx))
    plt.scatter(bracket, f_bracket, c="r")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Testing initial bracketing")
    plt.savefig("figures/tests/01_bracketing.png", dpi=600)


def test_find_min():
    from helperscripts.optimize import golden_section
    from helperscripts.prettyprint import pretty_print_timeit

    # Quadratic function with minimum -3 at x=0
    func = lambda x: -3 + np.array(x) ** 2
    xx = np.linspace(-5, 5, 1000)

    initial_bracket = (-2, 2)
    minimum = golden_section(func, *initial_bracket, atol=1e-12)
    print(f"Minimum found at {minimum:.2e}. Expected: 0.")

    plt.figure()
    plt.plot(xx, func(xx))
    plt.scatter(minimum, func(minimum), c="r", label="Found using golden_section")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Testing golden section search for minimum")
    plt.savefig("figures/tests/02_minimum.png", dpi=600)


def test_levenberg():
    from helperscripts.optimize import levenberg_marquardt
    from helperscripts.likelihoods import gaussian_logL, gaussian_logL_gradient
    import numpy as np

    xdata = np.linspace(1e-3, 5, 50)
    xplot = np.linspace(1e-3, 5, 1000)

    ## True model ##
    def model(x, a, b):
        return (a / x) ** 2 * np.exp(-b / x)

    ## Derivatives ##
    def d_model_da(x, a, b):
        return 2 * a / x**2 * np.exp(-b / x)

    def d_model_db(x, a, b):
        return -1 / x * np.exp(-b / x) * (a / x) ** 2

    p_true = [2, 1]

    # Noisy data
    y_true = model(xdata, *p_true)
    sigma = 0.1
    y_noisy = y_true + np.random.normal(0, sigma, xdata.shape)

    data = [xdata, y_noisy]

    ## Levenberg-Marquardt fitting procedure ##
    params = levenberg_marquardt(
        data=data,
        model=model,
        sigma=lambda x, *p: np.ones_like(x) * sigma,
        derivatives=(d_model_da, d_model_db),
        logL=gaussian_logL,
        dlogL_dp=gaussian_logL_gradient,
        p0=(1, 1),
        step=1e-3,
        weight=10,
        max_iters=10000,
        atol=0.01,
    )

    print(f"Best params: {params}")
    print(f"True params: {p_true}")

    plt.figure()
    plt.scatter(xdata, y_noisy, c="k", label="Noisy data")
    plt.plot(xplot, model(xplot, *p_true), c="r", label="True model")
    plt.plot(xplot, model(xplot, *params), c="blue", ls="--", label="Best fit")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("figures/tests/03_levenberg.png", bbox_inches="tight", dpi=600)


def test_quasi_newton():
    from helperscripts.optimize import quasi_newton
    from helperscripts.likelihoods import gaussian_logL, gaussian_logL_gradient
    import numpy as np

    xdata = np.linspace(1e-3, 5, 50)
    xplot = np.linspace(1e-3, 5, 1000)

    ## True model ##
    def model(x, a, b):
        return (a / x) ** 2 * np.exp(-b / x)

    ## Derivatives ##
    def d_model_da(x, a, b):
        return 2 * a / x**2 * np.exp(-b / x)

    def d_model_db(x, a, b):
        return -1 / x * np.exp(-b / x) * (a / x) ** 2

    p_true = [2, 1]

    # Noisy data
    y_true = model(xdata, *p_true)
    sigma = 0.1
    y_noisy = y_true + np.random.normal(0, sigma, xdata.shape)

    data = [xdata, y_noisy]

    ## Levenberg-Marquardt fitting procedure ##
    params = quasi_newton(
        data=data,
        model=model,
        sigma=lambda x, *p: np.ones_like(x) * sigma,
        derivatives=(d_model_da, d_model_db),
        logL=gaussian_logL,
        dlogL_dp=gaussian_logL_gradient,
        p0=(1, 0.5),
        max_iters=100,
        atol=1e-5,
    )

    print(f"Best params: {params}")
    print(f"True params: {p_true}")

    plt.figure()
    plt.scatter(xdata, y_noisy, c="k", label="Noisy data")
    plt.plot(xplot, model(xplot, *p_true), c="r", label="True model")
    plt.plot(xplot, model(xplot, *params), c="blue", ls="--", label="Best fit")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("figures/tests/04_newton.png", bbox_inches="tight", dpi=600)

def test_uniform_generator():
    """
    Test if the random generation works
    as intended
    """
    from helperscripts.random import Random
    from helperscripts.stat import pearson
    from helperscripts.prettyprint import pretty_print_timeit
    from timeit import timeit
    from numpy.random import uniform

    # Get uniformly distributed points
    generator = Random()
    points = generator.uniform(size=10_000)
    points_np = uniform(size=10_000)

    # Average and std of generated points
    avg = points.mean()  # Expected: 0.5
    var = points.var()  # Expected: 1/12
    avg_np = points_np.mean()  # Expected: 0.5
    var_np = points_np.var()  # Expected: 1/12

    # Correlation between successive numbers
    x = points[1:]
    y = points[:-1]
    corr = pearson(x, y)

    xnp = points_np[1:]
    ynp = points_np[:-1]
    corr_np = pearson(xnp, ynp)

    print("Own RNG")
    print(f"    Average of generated points: {avg}. Expected: 0.5")
    print(f"    Variance of generated points: {var}. Expected: 1/12 = 0.08333...")
    print(f"    Correlation between successive numbers: {corr}")

    print("\nNumpy RNG")
    print(f"    Average of generated points: {avg_np}. Expected: 0.5")
    print(f"    Variance of generated points: {var_np}. Expected: 1/12 = 0.08333...")
    print(f"    Correlation between successive numbers: {corr_np}")

    # Time the generation
    repeats = 10
    uniform_time = timeit(lambda: generator.uniform(size=100_000), number=repeats)
    uniform_time_np = timeit(lambda: uniform(size=100_000), number=repeats)

    print("\nOwn RNG")
    pretty_print_timeit(uniform_time, repeats, indented=1)
    print("Numpy RNG")
    pretty_print_timeit(uniform_time_np, repeats, indented=1)


def test_normal_generator():
    """
    Test if the random generation works
    as intended
    """
    from helperscripts.random import Random
    from helperscripts.stat import pearson
    from numpy.random import normal

    # Get uniformly distributed points
    generator = Random()
    points = generator.normal(0, 1, size=10_000)
    points_np = normal(0, 1, size=10_000)

    # Average and std of generated points
    avg = points.mean()  # Expected: 0
    var = points.var()  # Expected: 1
    avg_np = points_np.mean()  # Expected: 0
    var_np = points_np.var()  # Expected: 1

    # Correlation between successive numbers
    x = points[1:]
    y = points[:-1]
    corr = pearson(x, y)

    xnp = points_np[1:]
    ynp = points_np[:-1]
    corr_np = pearson(xnp, ynp)

    print("Own RNG")
    print(f"    Average of generated points: {avg}. Expected: 0")
    print(f"    Variance of generated points: {var}. Expected: 1")
    print(f"    Correlation between successive numbers: {corr}")

    print("\nNumpy RNG")
    print(f"    Average of generated points: {avg_np}. Expected: 0")
    print(f"    Variance of generated points: {var_np}. Expected: 1")
    print(f"    Correlation between successive numbers: {corr_np}")

    edges = np.linspace(-5, 5, 50)
    points_hist = np.histogram(points, bins=edges, density=True)[0]
    points_np_hist = np.histogram(points_np, bins=edges, density=True)[0]

    def analytic_gaussian(x, mu, sigma):
        return (
            1
            / np.sqrt(2 * np.pi * sigma**2)
            * np.exp(-0.5 * ((mu - x) / sigma) ** 2)
        )

    xx = np.linspace(-5, 5, 1000)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(title="Testing normal distribution", xlabel="x", ylabel="PDF")
    ax.plot(
        xx,
        analytic_gaussian(xx, 0, 1),
        c="gray",
        lw=5,
        alpha=0.7,
        label="Analytic normal PDF",
    )
    ax.stairs(points_hist, edges=edges, label="Own RNG", ec="r")
    ax.stairs(points_np_hist, edges=edges, label="Numpy RNG", ec="k", ls="--")
    ax.legend()

    fig.savefig("figures/tests/05_normal_dist.png", bbox_inches="tight", dpi=600)


def test_rng_multidim():
    from helperscripts.random import Random

    generator = Random()
    uniform = np.random.uniform(size=(10_000, 3))
    x = uniform[:, 0]
    y = uniform[:, 1]
    z = uniform[:, 2]

    print("Generated points in 3 dimensions")
    print("Statistics in x")
    print(f"    Average: {x.mean()}. Expected: 0.5")
    print(f"    Variance: {x.var()}. Expected: 1/12 = 0.08333...")
    print("Statistics in y")
    print(f"    Average: {y.mean()}. Expected: 0.5")
    print(f"    Variance: {y.var()}. Expected: 1/12 = 0.08333...")
    print("Statistics in z")
    print(f"    Average: {z.mean()}. Expected: 0.5")
    print(f"    Variance: {z.var()}. Expected: 1/12 = 0.08333...")
    print("Statistics over whole array")
    print(f"    Average: {uniform.mean()}. Expected: 0.5")
    print(f"    Variance: {uniform.var()}. Expected: 1/12 = 0.08333...")


def main():
    from helperscripts.prettyprint import pretty_print_title

    pretty_print_title("Now testing bracketing")
    test_bracketing()

    print()
    pretty_print_title("Now testing minimization")
    test_find_min()

    print()
    pretty_print_title("Now testing Levenberg")
    test_levenberg()
    
    print()
    pretty_print_title("Now testing quasi-newton BFGS")
    test_quasi_newton()

    print()
    pretty_print_title("Now testing normal number generation")
    test_normal_generator()

    print()
    pretty_print_title("Now testing uniform number generation")
    test_uniform_generator()

    print()
    pretty_print_title("Now testing mutli-dimensional uniform rng")
    test_rng_multidim()


if __name__ in ("__main__"):
    main()
