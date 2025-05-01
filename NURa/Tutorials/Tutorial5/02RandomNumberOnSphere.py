import numpy as np
import matplotlib.pyplot as plt


def generate_random_sphere(N: int, r: float = 1.0, method="uniform") -> np.ndarray:
    """
    Function that randomly generates N points on a sphere of
    radius r.

    Parameters
    ----------
    N : int
        Number of points
    r : float
        Radius of sphere
    method : str
        Method to use for random generation

    Returns
    -------
    (x, y, z): ndarrays
        Tuple with coordinates of random points on a sphere
    """
    if method == "uniform":
        theta = np.pi * np.random.uniform(0.0, 1.0, size=N)
        phi = 2 * np.pi * np.random.uniform(0.0, 1.0, size=N)
    elif method == "acos":
        theta = np.acos(1 - 2 * np.random.uniform(0.0, 1.0, size=N))
        phi = 2 * np.pi * np.random.uniform(0.0, 1.0, size=N)
    else:
        raise ValueError(f"Method {method} not recognized")

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return (x, y, z)


def main():
    # Number of random points
    N = 5000
    x, y, z = generate_random_sphere(N)

    ##############
    ## Plotting ##
    ##############
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    axticks = [-1, -0.5, 0, 0.5, 1]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xticks(axticks)
    ax.set_yticks(axticks)
    ax.set_zticks(axticks)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.scatter(x, y, z, c="b", s=4)

    fig.savefig("figures/02_random_sphere_Q2a.png", bbox_inches="tight", dpi=300)

    #################
    ## acos method ##
    #################
    x, y, z = generate_random_sphere(N, method="acos")

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    axticks = [-1, -0.5, 0, 0.5, 1]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xticks(axticks)
    ax.set_yticks(axticks)
    ax.set_zticks(axticks)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.scatter(x, y, z, c="b", s=4)

    fig.savefig("figures/02_random_sphere_Q2b.png", bbox_inches="tight", dpi=300)


if __name__ in ("__main__"):
    main()
