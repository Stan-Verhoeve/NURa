import numpy as np

class interpolator:
    def __init__(self, x_values: list | np.ndarray, y_values: list | np.ndarray) -> None:
        """
        Initialize with known x- and y-values

        Parameters
        ----------
        x_values: list | ndarray
            Known nodes for interpolation. Must be strictly monotonic
        y_values: list | ndarray
            Corresponding y-values of the x-nodes
        """
        assert interpolator.__strictly_monotonic(x_values), "`x_values` should be strictly monotonic"
        self.__x_values = x_values
        self.__y_values = y_values
        self.coefs = self._calc_linear_coefs()

    @property
    def x_values(self):
        return self.__x_values

    @property
    def y_values(self):
        return self.__y_values
    
    def _calc_linear_coefs(self):
        """
        Calculate coefficients for linear interpolation scheme
        """
        slopes = [(self.y_values[i] - self.y_values[i-1]) / (self.x_values[i] - self.x_values[i-1]) for i in range(1, len(self.x_values))]
        return slopes
    
    def _neville_interpolator(self, x: float, order: int = None, error_estimate: bool = False) -> float:
        """
        Polynomial interpolator based on Neville's algorithm

        Parameters
        ----------
        x : float
            Point to interpolate
        order : int
            Order of polynomial
        error_estimate : bool, optional
            Whether to give a first-order estimate of the error wrt the true function.
            The default is false

        Returns
        -------
        P[0] : float
            Interpolated value at x
        """
        if not order:
            order = len(self.x_values) - 1

        # order = M - 1, where M the number of points to consider
        M = order + 1

        # Find M closest points
        idx_mid = interpolator.bisection(x, self.x_values)
        
        # Get lower index, starting at 0
        low = max(0, idx_mid - (M // 2))
        # Get higher index
        high = min(low + M, len(self.y_values))
        
        if high - low < M:
            low = high - M
        
        # Matrix to store coefficients
        P = self.y_values[low:high].copy()
        
        # Iterate over kernels
        for k in range(1,M):
            for i in range(M-k):
                j = i + k
                
                # Overwrite if necessary
                P[i] = ((self.x_values[low:high][j] - x) * P[i] + (x - self.x_values[low:high][i]) * P[i+1]) / (self.x_values[low:high][j] - self.x_values[low:high][i])

        
        if error_estimate:
            return P[0], P[1]

        return P[0]

    def _linear_interpolator(self, x: list | np.ndarray) -> float:
        """
        Linear interpolator

        Parameters
        ----------
        x: float
            Point to interpolate

        Returns
        -------
        y_interp : float
            Interpolated value at x
        """
        # If x in nodes, return known node value
        if x in self.x_values:
            idx = np.where(self.x_values == x)
            return self.y_values[idx][0]
        
        # Find injection index
        idx = interpolator.bisection(x, self.x_values) - 1
        
        # Grab precomputed slope
        slope = self.coefs[idx]
        y_interp = slope * (x - self.x_values[idx]) + self.y_values[idx]

        return y_interp

    def _calc_cubic_coefs(self, order: int = 3) -> None:
        """
        Calculate coefficients for cubic interpolation scheme using Neville's algorithm
        """
        # TODO: see if it is possible to precompute Neville coefficients
        NotImplemented

        

    def interpolate(self, x: list | np.ndarray, kind: str = "linear") -> np.ndarray:
        """
        Interpolate on the values `x`

        Parameters
        ---------
        x: float | list | ndarray
            Values to interpolate on
        kind: str
            Interpolation kind. Can be "linear" or "cubic" or "neville".
            If "neville", will interpolate on (N-1)-degree polynomial, where N
            is the number of nodes

        Returns
        -------
        y_interp: ndarray
            Interpolated values at `x`

        """
        assert self.__bounded(x), "`x` is out of bounds for the given `x_values`"
        
        x = np.array(x)
        y_interp = np.zeros_like(x)
        
        # Grab correct interpolation function
        match kind:
            case "linear":
                interp_func = self._linear_interpolator
            case "cubic":
                interp_func = lambda x: self._neville_interpolator(x, order=3)
            case "neville":
                interp_func = lambda x: self._neville_interpolator(x)
        
        # Iterate over points to interpolate
        for i, xi in enumerate(x):
            y_interp[i] = interp_func(xi)

        return y_interp


    @staticmethod
    def __strictly_monotonic(array: list | np.ndarray) -> bool:
        """
        Check if an array is strictly monotonic

        Parameters
        ----------
        array : list | np.ndarray
            Array to check

        Returns
        -------
        bool
            True if strictly monotonic, false if not
        """
        strictly_increasing = all(a < b for a, b in zip(array, array[1:]))
        strictly_decreasing = all(a > b for a, b in zip(array, array[1:]))

        return strictly_increasing | strictly_decreasing

    def __bounded(self, x: list | np.ndarray) -> bool:
        """
        Check if a value is bounded to the domain of `x_values`

        Parameters
        ----------
        x: list | ndarray
            Values to check for boundedness

        Returns
        -------
        bool
            True if bounded, false if not
        """
        return all(x >= min(self.x_values)) & all(x <= max(self.x_values))

    @staticmethod
    def bisection(a, array: list | np.ndarray) -> int:
        """
        Use bisection algorithm to find where in array `a` should
        be. If `a` is in `array`, will return that index

        Parameters
        ----------
        a : float
            Value to insert
        array : list | ndarray
            Array to check where to insert

        Returns
        -------
        low: int
            Index where to insert `a`
        """
        # Bounds start at edge of array
        low = 0
        high = len(array)

        while low < high:
            halfway = (low + high) // 2
            
            # Check if array is on left
            left = a < array[halfway]
            if left:
                # Change high end to halfway point
                high = halfway
            # If not on left, change low end to halfway point
            else:
                low = halfway + 1

        return low
