import numpy as np


def polynomial(x, coefs):
    if len(coefs) == 1:
        return np.array(coefs)[0]

    return np.array(coefs)[0] + x * polynomial(x, coefs[1:])


def polynomial_broad(x, coefs):
    powers = np.arange(len(coefs))
    x = np.atleast_1d(x)
    return np.dot(x[:, None] ** powers, coefs)


def test_polynomial_speed():
    from timeit import timeit
    from helperscripts.prettyprint import pretty_print_timeit

    x = np.linspace(-10, 10, 1000)
    coefs = [5, 4, 3, 2, 1]

    poly1 = polynomial(x, coefs)
    poly2 = polynomial(x, coefs)

    print(f"Polynomials equal: {np.all(np.isclose(poly1, poly2))}")
    Nloops = 10_000

    recursive = timeit(lambda: polynomial(x, coefs), number=Nloops)
    broadcast = timeit(lambda: polynomial_broad(x, coefs), number=Nloops)

    print("Polynomial using recursion")
    pretty_print_timeit(recursive, Nloops, units="us", indented=1)
    print("\nPolynomial using broadcasting")
    pretty_print_timeit(broadcast, Nloops, units="us", indented=2)


def test_bracketing():
    """
    Quick test on finding a bracket for a function
    """
    from helperscripts.optimize import find_bracket

    # Quadratic function with minimum -3 at x=0
    func = lambda x: polynomial(x, (-3, 0, 1))
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

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(xx, func(xx))
    plt.scatter(bracket, f_bracket, c="r")
    plt.savefig("figures/tests/bracketing.png", dpi=600)


def test_find_min():
    from helperscripts.optimize import golden_section, golden_section_gpt
    from helperscripts.prettyprint import pretty_print_timeit

    # Quadratic function with minimum -3 at x=0
    func = lambda x: polynomial(x, (-3, 0, 1))
    xx = np.linspace(-5, 5, 1000)

    initial_bracket = (-2, 2)
    minimum = golden_section(func, *initial_bracket, atol=1e-12)
    print(f"Minimum found at {minimum:.2e}. Expected: 0.")

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(xx, func(xx))
    plt.scatter(minimum, func(minimum), c="r", label="Found using golden_section")
    plt.savefig("figures/tests/minimum.png", dpi=600)


def test_levenberg():
    from helperscripts.optimize import levenberg_marquardt
    import numpy as np

    xdata = np.linspace(1e-3, 5, 10)
    xplot = np.linspace(1e-3, 5, 1000)

    ## True model ##
    def model(x, a, b):
        return (a / x) ** 2 * np.exp(-b / x)

    ## Derivatives ##
    def d_model_da(x, a, b):
        return 2 * a / x**2 * np.exp(-b / x)

    def d_model_db(x, a, b):
        return -1 / x * np.exp(-b / x) * (a / x) ** 2

    def gauss_grad(data, model, sigma, derivatives, p):
        x, y = data[:, 0], data[:, 1]
        f = model(x, *p)

        # Jacobian
        J = [df(x, *p) for df in derivatives]
        J = np.stack(J, axis=1)
        res = (y - f) / sigma**2

        return -2 * J.T @ res

    def logL(data, model, sigma, p):
        x, y = data[:, 0], data[:, 1]
        res = y - model(x, *p)
        return np.sum(res**2 / sigma**2)

    p_true = [2, 1]

    # Noisy data
    sigma = 0.1
    y_noisy = model(xdata, *p_true) + np.random.normal(0, sigma, xdata.shape)

    data = np.stack([xdata, y_noisy], axis=1)

    ## Levenberg-Marquardt fitting procedure ##
    params = levenberg_marquardt(
        data=data,
        model=model,
        sigma=sigma,
        derivatives=(d_model_da, d_model_db),
        logL=logL,
        dlogL_dp=gauss_grad,
        p0=(1, 1),
        step=1e-3,
        weight=10,
        max_iters=10000,
        atol=0.01,
    )

    print(f"Best params: {params}")
    print(f"True params: {p_true}")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(*data.T, c="k", label="Noisy data")
    plt.plot(xplot, model(xplot, *p_true), c="r", label="True model")
    plt.plot(xplot, model(xplot, *params), c="blue", ls="--", label="Best fit")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("figures/tests/levenberg.png", bbox_inches="tight", dpi=600)


def test_random_generator():
    """
    Test if the random generation works
    as intended
    """
    from helperscripts.random import Random, pearson
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

    print("Own RNG")
    pretty_print_timeit(uniform_time, repeats, indented=1)
    print("Numpy RNG")
    pretty_print_timeit(uniform_time_np, repeats, indented=1)


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


def test_integration():
    """
    Tests if the integration scheme works
    as intended
    """
    from numpy import isclose
    from helperscripts.integrate import romberg, MCintegrator
    from helperscripts.prettyprint import pretty_print_timeit
    from timeit import timeit

    # Functions to integrate
    known_closed = lambda x: -(x**2)
    known_open = lambda x: x ** (-0.5)

    bounds = (0, 1)
    closed_romberg = romberg(known_closed, bounds, m=15)  # Analytic result: -1/3
    open_romberg = romberg(known_open, bounds, m=15)  # Analytic result: 2
    closed_MC = MCintegrator(known_closed, bounds)
    open_MC = MCintegrator(known_open, bounds)

    # Time the approaches
    repeats = 10
    closed_romberg_time = timeit(
        lambda: romberg(known_closed, bounds, m=15), number=repeats
    )
    closed_MC_time = timeit(
        lambda: MCintegrator(known_closed, bounds), number=repeats
    )
    open_romberg_time = timeit(
        lambda: romberg(known_open, bounds, m=15), number=repeats
    )
    open_MC_time = timeit(lambda: MCintegrator(known_open, bounds), number=repeats)

    print("Closed function x^2 (expected: -1/3)")
    print(f"    Romberg integration: {closed_romberg}")
    pretty_print_timeit(closed_romberg_time, repeats, indented=2)
    print(f"    MC integration: {closed_MC}")
    pretty_print_timeit(closed_MC_time, repeats, indented=2)
    print()
    print("Half-open function 1/sqrt(x) (expected: 2)")
    print(f"    Romberg integration: {open_romberg}")
    pretty_print_timeit(open_romberg_time, repeats, indented=2)
    print(f"    MC integration: {open_MC}")
    pretty_print_timeit(open_MC_time, repeats, indented=2)


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
    # pretty_print_title("Now testing random number generation")
    # test_random_generator()

    print()
    # pretty_print_title("Now testing mutli-dimensional rng")
    # test_rng_multidim()

    print()
    # pretty_print_title("Now testing integration")
    # test_integration()

    # print()
    # pretty_print_title("Now testing polynomial speed")
    # test_polynomial_speed()


if __name__ in ("__main__"):
    main()
