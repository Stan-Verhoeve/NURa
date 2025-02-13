import timeit

def main():
    # Setup for timeit
    setup = (
    "import numpy as np\n"
    "# Define constants\n"
    "c = 299_792_458  # m/s\n"
    "c_inv = 1/c\n"
    "c_inv2 = c_inv * c_inv\n"
    "G = 6.6743e-11  # m3 / kg / s2\n"
    "\n"
    "# Number of masses to draw\n"
    "N = 10_000\n"
    "\n"
    "# Draw masses\n"
    "masses = np.random.normal(loc=1e6, scale=1e5, size=N)\n"
    )
    
    # Number or repeats for timeit
    N = 50_000

    # Runs for timeit
    direct_run = "2 * G * masses / c ** 2"
    predefined_run = "2 * G * masses * c_inv2"
    
    time_direct = timeit.timeit(direct_run, setup=setup, number=N)
    time_predefined = timeit.timeit(predefined_run, setup=setup, number=N)

    # Calculate Schwarzschild radius and time it
    print(f"Average time direct use of equation: {time_direct / N * 1e6:.3f} usec")
    print(f"Average time having pre-defined c_inv2: {time_predefined / N * 1e6:.3f} usec")

if __name__ in ("__main__"):
    main()
