import numpy as np
import timeit


def polygon_area(x: list, y: list) -> float:
    """
    Function that calculates area of a simple
    polygon given the vertices (x,y)

    Parameters
    ----------
    x: list
        x-coordinates of the vertices
    y: list
        y-coordinates of the vertices

    Returns
    -------
    area: float
        Area of the simple polygon
    """
    assert len(x) == len(y), "`x` and `y` should have the same length"

    # Number of elements
    N = len(x)

    # Area starts at zero
    area = 0.0

    # Iterate over elements
    for i in range(N - 1):
        area += x[i] * y[i + 1] - x[i + 1] * y[i]

    return 0.5 * abs(area)


def polygon_area_vector(x: list, y: list) -> float:
    """
    Function that calculates area of a simple
    polygon given the vertices (x,y). Uses
    numpy vectorisation

    Parameters
    ----------
    x: list
        x-coordinates of the vertices
    y: list
        y-coordinates of the vertices

    Returns
    -------
    area: float
        Area of the simple polygon
    """

    # Shifted x- and y arrays
    x_shifted = x.copy()
    y_shifted = y.copy()

    # Remove first element, add it to end of array
    # effectively rolling the array to the right by
    # one element
    x_shifted.append(x_shifted.pop(0))
    y_shifted.append(y_shifted.pop(0))

    # Use np.dot for inner product, avoiding for-loop
    area = 0.5 * abs(np.dot(x, y_shifted) - np.dot(x_shifted, y))

    return area


def main():
    # Number of runs for timing
    N = 100_000

    # Vertices of simple polygon
    setup = "x_vertices = [0, 1, 2, 1]\n" "y_vertices = [0, 3, 4, 3]"

    # Time the two approaches
    runtime_scalar = timeit.timeit(
        "polygon_area(x_vertices, y_vertices)",
        setup=setup,
        globals=globals(),
        number=N,
    )
    runtime_vector = timeit.timeit(
        "polygon_area_vector(x_vertices, y_vertices)",
        setup=setup,
        globals=globals(),
        number=N,
    )

    print(f"Average scalar runtime: {runtime_scalar / N * 1e6:.3f} usec")
    print(f"Average vector runtime: {runtime_vector / N * 1e6:.3f} usec")


if __name__ in ("__main__"):
    main()
