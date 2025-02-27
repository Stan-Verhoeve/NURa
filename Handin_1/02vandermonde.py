import numpy as np
import copy


class LU_decomposition:
    def __init__(self, matrix: np.ndarray) -> None:
        """
        Class to perform LU decomposition with partial
        (implicit) pivoting

        Parameters
        ----------
        matrix : ndarray
            matrix to perform LU decomposition on
        """
        # Explicitly cast to float63 ndarray
        self.matrix = np.array(matrix, dtype=np.float64)
        self.LU = copy.deepcopy(self.matrix)

        # Confirm matrix is square
        if not LU_decomposition.__is_square(self.LU):
            raise ValueError("Matrix should be square")

        self.permutation = np.arange(len(self.LU))
        # self.permutation = np.zeros(len(self.LU), dtype=np.int32)
        self._decompose()

    def _decompose(self):
        """
        Performs LU decomposition in-place
        """
        largest_coef = np.max(np.abs(self.LU), axis=1)
        if not all(largest_coef > 0):
            raise ValueError("Matrix is singular")
        largest_coef_inv = largest_coef ** (-1)
        N = len(self.LU)

        # Iterate over columns
        for k in range(N):
            # Index of largest pivot
            imax = k + np.argmax(np.abs(self.matrix[k:, k] * largest_coef_inv[k:]))

            # Swap rows if imax not on diagonal
            if imax != k:
                self.LU[[imax, k], :] = self.LU[[k, imax], :]
                self.permutation[k], self.permutation[imax] = (
                    self.permutation[imax],
                    self.permutation[k],
                )
                largest_coef_inv[k], largest_coef_inv[imax] = (
                    largest_coef_inv[imax],
                    largest_coef_inv[k],
                )

            for i in range(k + 1, N):
                self.LU[i, k] /= self.LU[k, k]
                self.LU[i, k + 1 :] -= np.dot(self.LU[i, k], self.LU[k, k + 1 :])

        # idx_array stores the permutation destinations. For numpy indexing,
        # however, we want the inverse permutation that has in idx0 the row that
        # should go to 0, not the one that comes from zero.
        # As such, invert the permutation
        self.inv_permutation = np.empty_like(self.permutation)
        self.inv_permutation[self.permutation] = np.arange(N)

    def get_LU(self, separate=False):
        """
        Returns the decomposition, along with
        the permutation vector

        Parameters
        ----------
        separate : bool
            Separate LU into L and U.
            The default is false

        Returns
        -------
        tuple
            (LU, permutation) if separate=False
            (L, U, permutation) if separate=True
        """
        if separate:
            L = np.tril(self.LU, k=-1) + np.eye(len(self.LU))
            U = np.triu(self.LU)
            return (L, U, self.inv_permutation)
        return (self.LU, self.inv_permutation)

    @staticmethod
    def __is_square(matrix: np.ndarray) -> bool:
        """
        Checks if a given matrix is square

        Parameters
        ----------
        matrix : np.ndarray
            Matrix to check

        Returns
        -------
        bool
            Whether matrix is square
        """

        # Convert shape to set (i.e. get unique elements)
        # If square, there is 1 unique element in the set
        return len(set(np.shape(matrix))) == 1


def forward_substitution(L, y):
    N = len(y)
    z = np.zeros(N)

    for i in range(N):
        z[i] = y[i] - np.dot(L[i, :i], z[:i])
    return z


def backward_substitution(U, y):
    N = len(y)
    z = np.zeros(N)

    for i in range(N)[::-1]:
        z[i] = (y[i] - np.dot(U[i, i:], z[i:])) / U[i, i]
    return z


def polynomial(coefs, x):
    y = 0.0
    for power, c in enumerate(coefs):
        y += c * x**power

    return y


def solve_system(M: np.ndarray, y: np.ndarray, Niters: int = None) -> np.ndarray:
    """
    Solve the matrix system Mx=y.

    Parameters
    ----------
    M : list | ndarray
        Matrix of the system of equations
    y : list | ndarray
        Solution vector
    Niters : int, optional
        Number of iterations for iterative improvement.
        If none, does not use iterative improvement.
        The default is None

    Returns
    -------
    x : ndarray
        Solution to the system Mx=y
    """
    # LU decomposition
    decomposition = LU_decomposition(M)
    L, U, permutation = decomposition.get_LU(separate=True)

    # Intermediate solution vector
    z = forward_substitution(L, y[permutation])
    # Solution
    x = backward_substitution(U, z)

    if Niters:
        for _ in range(Niters):
            dy = M @ x - y

            # Intermediate solution
            z = forward_substitution(L, dy)
            # Error in c
            dx = backward_substitution(U, z)
            # Subtract to minimise
            x -= dx

    return x


def main():
    from helperscripts.interpolation import interpolator
    from helperscripts.prettyprint import (
        pretty_print_array,
        pretty_print_timeit,
    )
    import matplotlib.pyplot as plt
    import timeit

    # Get x, y data
    x, y = np.genfromtxt("Vandermonde.txt").T

    # Construct Vandermonde matrix
    V = x[:, None] ** np.arange(len(x))

    # LU decomposition
    decomposition = LU_decomposition(V)
    L, U, permutation = decomposition.get_LU(separate=True)

    # Sanity check
    # Include permutation to undo the row swaps
    reconstruction = (L @ U)[permutation, :]

    print("Sanity check")
    print(f"LU = V: {np.all(np.isclose(V, reconstruction))}\n")

    ##########################
    ## Solving coefficients ##
    ##########################

    coefs = solve_system(V, y)
    print("Polynomial coefficients:")
    pretty_print_array(coefs, ncols=4)

    # Solve with iterative correction
    coefs_iter1 = solve_system(V, y, Niters=1)
    coefs_iter10 = solve_system(V, y, Niters=10)

    ###############################################
    ## Interpolation and polynomial construction ##
    ###############################################
    # Create interpolator object
    ipl = interpolator(x, y)

    # Points to interpolate on
    interp_x = np.linspace(min(x), max(x), 1000)

    # Interpolate using Neville
    neville_y = ipl.interpolate(interp_x, kind="neville")

    # Get polynomial using Vandermonde coefficients
    LU_y = polynomial(coefs, interp_x)
    LU_y_iter1 = polynomial(coefs_iter1, interp_x)
    LU_y_iter10 = polynomial(coefs_iter10, interp_x)

    # Get absolute differences
    abs_diff_neville = abs(y - ipl.interpolate(x, kind="neville"))
    abs_diff_LU = abs(y - polynomial(coefs, x))
    abs_diff_LU_iter1 = abs(y - polynomial(coefs_iter1, x))
    abs_diff_LU_iter10 = abs(y - polynomial(coefs_iter10, x))

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

    ax1.set_ylim(-400, 400)
    ax1.set_ylabel("y")
    ax2.set_xlabel("x")
    ax2.set_ylabel("|$y_i - y$|")
    ax2.set_yscale("log")

    # Q2a
    ax1.scatter(x, y, label="Nodes")
    ax1.plot(interp_x, LU_y, c="orange", label="LU decomposition")
    ax2.plot(x, abs_diff_LU, c="orange")
    ax1.legend()
    plt.savefig("figures/02_vandermonde_Q2a.png", bbox_inches="tight", dpi=300)

    # Q2b
    ax1.plot(interp_x, neville_y, c="green", ls="--", label="Neville")
    ax2.plot(x, abs_diff_neville, c="green", ls="--")
    ax1.legend()
    plt.savefig("figures/02_vandermonde_Q2b.png", bbox_inches="tight", dpi=300)

    # Q2c
    ax1.plot(interp_x, LU_y_iter1, c="red", ls="dotted", label=f"LU, 1 iterations")
    ax1.plot(interp_x, LU_y_iter10, c="purple", ls="-.", label=f"LU, 10 iterations")
    ax1.legend()
    ax2.plot(x, abs_diff_LU_iter1, c="red", ls="dotted")
    ax2.plot(x, abs_diff_LU_iter10, c="purple", ls="-.")

    plt.savefig("figures/02_vandermonde_Q2c.png", bbox_inches="tight", dpi=300)

    ##############
    ##  Timeit  ##
    ##############
    number = 100
    time_neville = timeit.timeit(
        lambda: ipl.interpolate(interp_x, kind="neville"), number=number
    )
    time_LU = timeit.timeit(lambda: solve_system(V, y), number=number)
    time_LU_iter1 = timeit.timeit(
        lambda: solve_system(V, y, Niters=1), number=number
    )
    time_LU_iter10 = timeit.timeit(
        lambda: solve_system(V, y, Niters=10), number=number
    )

    print("\nNeville interpolation:")
    pretty_print_timeit(time_neville, number)

    print("\nLU decomposition: (no iterations)")
    pretty_print_timeit(time_LU, number)

    print("\nLU decomposition (1 iteration)")
    pretty_print_timeit(time_LU_iter1, number)

    print("\nLU decomposition (10 iterations)")
    pretty_print_timeit(time_LU_iter10, number)


if __name__ in ("__main__"):
    main()
