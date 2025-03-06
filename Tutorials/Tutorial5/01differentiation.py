import numpy as np
import matplotlib.pyplot as plt

def finite_difference(function, x, h):
    """
    Computes derivative using finite differences
    """
    h_inv = h ** (-1)
    dy = (function(x + h) - function(x - h)) * h_inv * 0.5

    return dy

def ridder(function, x, h_init, d, eps, max_iters=10):
    ridder_matrix = np.empty((max_iters, len(x)))
    d_arr = d ** 2*(np.arange(1,max_iters) )

    d_inv = d**(-1)
    
    # Populate ridder matrix
    for i in range(max_iters):
        ridder_matrix[i] = finite_difference(function, x, h_init * (d_inv ** i))
    
    previous = ridder_matrix[0].copy()
    for j in range(1,max_iters):
        factor = d_arr[j-1]

        ridder_matrix = (factor * ridder_matrix[1:] - ridder_matrix[:-1]) / (factor - 1)

        # for i in range(max_iters-j):
        #     ridder_matrix[i] = (d ** (2*j) * ridder_matrix[i+1] - ridder_matrix[i]) / (d**(2*j) - 1)
        error = abs(ridder_matrix[0] - previous)
        previous = ridder_matrix[0].copy()
        if np.all(error < eps):
            break

    if j == max_iters:
        print("Warning: maximum iterations reached!")

    
    df = ridder_matrix[0]
    
    # for a single x
    # D[:-1, j] = D[1:, j-1] - D[:-1, j-1]

    # for i in range(np.shape(D)[0]):
    return df

def main():
    # Function to differentiate: x^2 sin(x)
    function = lambda x: x**2 * np.sin(x)

    # Analytic derivative to x^2 sin(x)
    analytic_derivative_func = lambda x: 2 * x * np.sin(x) + x**2 * np.cos(x)
    
    # Range to plot on
    xrange = [0, 2*np.pi]
    plotx = np.linspace(*xrange, 200)
    analytic_derivative = analytic_derivative_func(plotx)

    ##############
    ## Plotting ##
    ##############
    aspect = 2
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        gridspec_kw={"height_ratios": [aspect, 1]},
        figsize=(8, 6),
        sharex=True,
        constrained_layout=True,
        dpi=300,
    )

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax2.set_ylabel(r"|$y_\text{analytic} - y_\text{numeric}$|")
    ax2.set_yscale("log")

    ax1.plot(plotx, analytic_derivative, c="gray", alpha=0.7, lw=5, label="Analytic derivative")
    ax1.grid()
    ax1.legend()
    
    fig.savefig("figures/01_differentiation_Q1a.png", bbox_inches="tight", dpi=300)
    
    linestyles = ["solid", "--", ":"]
    colours = ["green", "purple", "orange"]
    
    for i,h in enumerate([0.1, 0.01, 0.001]):
        numeric_derivative = finite_difference(function, plotx, h)

        abs_difference = abs(analytic_derivative - numeric_derivative)
        ax1.plot(plotx, numeric_derivative, c=colours[i], ls=linestyles[i], label=f"Finite difference with h={h}")
        ax2.plot(plotx, abs_difference, c=colours[i], ls=linestyles[i])

        ax1.legend()
    fig.savefig("figures/01_differentiation_Q1b.png", bbox_inches="tight", dpi=300)
    
    ############
    ## Ridder ##
    ############
    relative_error = 1e-4
    max_iters = 10

    ridder_derivative = ridder(function, plotx, 5, 2, relative_error, max_iters)
    abs_difference_ridder = abs(analytic_derivative - ridder_derivative)
    ax1.plot(plotx, ridder_derivative, c="k", ls="-.", label=f"Ridder, max_iters={max_iters}, relative_error={relative_error:.1e}")
    ax2.plot(plotx, abs_difference_ridder, c="k", ls="-.")
    ax1.legend()

    fig.savefig("figures/01_differentiation_Q1c.png", bbox_inches="tight", dpi=300)

if __name__ in ("__main__"):
    main()
