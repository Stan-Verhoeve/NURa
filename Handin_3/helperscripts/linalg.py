import numpy as np
import matplotlib.pyplot as plt
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


def forward_backward_substitution(LU, b):
    """
    Perform both forward and backward substitution using a combined LU matrix.
    - LU is the combined matrix with the lower part (L) and upper part (U).
    - b is the vector on the right-hand side.

    Returns the solution vector x.
    """
    N = len(b)
    z = np.zeros(N)

    # Perform forward substitution to solve Lz = b
    for i in range(N):
        # Calculate z[i] using the lower triangular part of LU
        z[i] = b[i] - np.dot(LU[i, :i], z[:i])

    # Perform backward substitution to solve Ux = z
    x = np.zeros(N)
    for i in range(N - 1, -1, -1):
        # Calculate x[i] using the upper triangular part of LU
        x[i] = (z[i] - np.dot(LU[i, i + 1 :], x[i + 1 :])) / LU[i, i]

    return x


def solve_system(M: np.ndarray, y: np.ndarray, Niters: int = None) -> np.ndarray:
    """
    Solve the matrix system Mx=y using LU decomposition stored in a single matrix LU.

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
    # LU decomposition (get the LU matrix)
    decomposition = LU_decomposition(M)
    LU, permutation = decomposition.get_LU()
    # Intermediate solution vector
    z = forward_backward_substitution(LU, y[permutation])
    # Solution x
    x = z  # Since z is the final solution after forward-backward substitution

    if Niters:
        for _ in range(Niters):
            dy = M @ x - y

            # Intermediate solution
            z = forward_backward_substitution(LU, dy)
            # Error in c
            dx = z
            # Subtract to minimize
            x -= dx

    return x
