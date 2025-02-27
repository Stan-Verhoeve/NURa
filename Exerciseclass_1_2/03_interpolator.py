from matplotlib.image import imread
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
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
    
    def _neville_interpolator(self, x: float, order: int, error_estimate: bool = False) -> float:
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
            Interpolation kind. Can be "linear" or "cubic"

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

def test():
    # Test function
    func = lambda x: np.sin(2 * np.pi * x)
    x_nodes = np.linspace(0, 1, 8)
    y_nodes = func(x_nodes)
    
    # Interpolator object
    ipl = interpolator(x_nodes, y_nodes)
    
    # Interpolated values
    x_want = np.linspace(0, 1, 1000)
    y_interp = ipl.interpolate(x_want)
    y_interp_cubic = ipl.interpolate(x_want, kind="cubic")

    # Create figure
    fig = plt.figure(dpi=200, figsize=(12, 4))
    ax = fig.add_subplot(111)
        
    # Original nodes
    ax.scatter(x_nodes, y_nodes, c="k", marker="x")

    # Interpolations
    ax.plot(x_want, y_interp, label="Linear interpolation", c="b")
    ax.plot(x_want, y_interp_cubic, label="Cubic interpolation", c="r", ls=":")

    # Analytic function
    ax.plot(x_want, func(x_want), c="gray", alpha=0.5, lw=6, label="Analytic")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    plt.show()

def main():
    # Read image and grab first row
    image = imread("M42_128.jpg").astype(float)
    first_row_values = image[0,:]
    first_row_pixels = np.arange(0,first_row_values.size)
    
    x_want = np.linspace(0, max(first_row_pixels), 201)
    
    # Interpolator object
    ipl = interpolator(first_row_pixels, first_row_values)
    
    # Interpolate values
    linear_interpolated = ipl.interpolate(x_want, kind="linear")
    cubic_interpolated = ipl.interpolate(x_want, kind="cubic")
    
    # Create figure
    fig = plt.figure(dpi=300, figsize=(9,6))
    ax = fig.add_subplot(111)
    ax.set_title("Interpolation of first row")

    # Plot original data
    ax.plot(first_row_pixels, first_row_values, c="gray", alpha=0.5, lw=6, label="Original")

    # Plot interpolations
    ax.plot(x_want, linear_interpolated, c="b", label="Linear interpolation")
    ax.plot(x_want, cubic_interpolated, c="r", label="Cubic interpolation", ls=":")

    ax.set_xlabel("Pixels")
    ax.set_ylabel("Intensity")
    ax.legend()
    plt.show()

    # TODO: Check if this is indeed the correct way to interpolate
    # in two dimensions

    # Get width and height of original image
    width, height = image.shape

    # Interpolate s.t. image has twice the resolution
    interp_width = int(2 * width)
    interp_height = int(2 * height)

    # Pixel coordinates of original image
    im_x = np.arange(0, image.shape[0])
    im_y = np.arange(0, image.shape[1])
    
    # (Pixel) coordinates of interpolated image
    interp_x = np.linspace(min(im_x), max(im_x), interp_width)
    interp_y = np.linspace(min(im_y), max(im_y), interp_height)

    # Array to store interpolation along x-axis
    interp_im_x = np.zeros((len(interp_x), len(im_y)))
    
    # Iterate over rows
    for i, yi in enumerate(im_y):
        # Interpolate along x
        ipl = interpolator(im_x, image[:,i])
        interpolated_row = ipl.interpolate(interp_x, kind="cubic")
        interp_im_x[:,i] = interpolated_row
    
    # Array to store interpolation along both axes
    interp_im_xy = np.zeros((len(interp_x), len(interp_y)))
    
    # Iterate over columns
    for j, xj in enumerate(interp_x):
        # Interpolate along y
        ipl = interpolator(im_y, interp_im_x[j, :])
        interpolated_column = ipl.interpolate(interp_y, kind="cubic")

        # Since we had already interpolated the rows, we do not need
        # to consider interpolation along x here; it is already
        # taken care of by construction
        interp_im_xy[j, :] = interpolated_column
    
    
    # Create figure
    fig, axs = plt.subplots(1,2, figsize=(24, 12))
    orig_ax, interp_ax = axs
    
    # Plot original and interpolated image
    orig_ax.imshow(image)
    interp_ax.imshow(interp_im_xy)
    
    orig_ax.set_title("Original")
    interp_ax.set_title("Cubic interpolated")

    plt.tight_layout()
    plt.show()

if __name__ in ("__main__"):
    # test()
    main()
