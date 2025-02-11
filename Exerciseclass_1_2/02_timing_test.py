# Use subprocess to run timeit in shell
import subprocess

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
    
    # Runs for timeit
    direct_run = "2 * G * masses / c ** 2"
    predefined_run = "2 * G * masses * c_inv2"
    
    # Calculate Schwarzschild radius and time it
    print("Direct use of equation:")
    subprocess.run(["python3", "-m", "timeit", "-n", "50000", "-s", setup, direct_run])

    print("\nHaving pre-defined c_inv2:")
    subprocess.run(["python3", "-m", "timeit", "-n", "50000", "-s", setup, predefined_run])

if __name__ in ("__main__"):
    main()
