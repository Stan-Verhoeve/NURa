from matplotlib.image import imread
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

image = imread("M42_128.jpg")

class interpolator:
    def __init__(self, x_values, y_values, kind="linear"):
        """
        Initialize with known x- and y-values
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
    
    def _calc_cubic_coefs(self):
        """
        Calculate coefficients for cubic interpolation scheme using Neville's algorithm
        """

        NotImplemented

    def interpolate(self, x):
        assert self.__bounded(x), "`x` is out of bounds for the given `x_values`"
        
        x = np.array(x)
        y_interp = np.zeros_like(x)

        for i, xi in enumerate(x):
            if xi in self.x_values:
                idx = np.where(self.x_values == xi)
                y_interp[i] = self.y_values[idx]
                continue
            
            # Find injection index
            idx = interpolator.bisection(xi, self.x_values) - 1
            
            # Grab precomputed slope
            slope = self.coefs[idx]
            y_interp[i] = slope * (xi - self.x_values[idx]) + self.y_values[idx]

        return y_interp


    @staticmethod
    def __strictly_monotonic(array):
        """
        Check if an array is strictly monotonic
        """
        strictly_increasing = all(a < b for a, b in zip(array, array[1:]))
        strictly_decreasing = all(a > b for a, b in zip(array, array[1:]))

        return strictly_increasing | strictly_decreasing

    def __bounded(self, x):
        """
        Check if a value is bounded to the domain of `x_values`
        """
        return all(x >= min(self.x_values)) & all(x <= max(self.x_values))

    @staticmethod
    def bisection(a, array):
        """
        Use bisection algorithm to find where in array `a` should
        be. If `a` is in `array`, will return that index

        Parameters
        ----------
        a : float
            Value to insert
        array : list | ndarray
            

        Find where in array x should be
        using bisection algorithm
        If a is equal to an element in a, will
        return the index of that element
        """
        low = 0
        high = len(array)

        while low < high:
            halfway = (low + high) // 2
            
            # Check if array is on right
            left = a < array[halfway]
            if left:
                # Change high end to halfway point
                high = halfway
            # If not on left, change low end to halfway point
            else:
                low = halfway + 1

        return low

def main():
    # Test function
    func = lambda x: np.sin(2 * np.pi * x * 2)
    x_nodes = np.linspace(0, 1, 10)
    y_nodes = func(x_nodes)
    
    # Interpolator object
    ipl = interpolator(x_nodes, y_nodes)
    
    # Interpolated values
    x_want = np.linspace(0, 1, 100)
    y_interp = ipl.interpolate(x_want)
    
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
        
    # Original nodes
    ax.scatter(x_nodes, y_nodes)

    # Interpolations
    ax.plot(x_want, y_interp, label="Linear interpolation")
    ax.plot(x_want, np.interp(x_want, x_nodes, y_nodes), ls="dotted", label="np.interp()")
    
    # Analytic function
    ax.plot(x_want, func(x_want), ls="--", label="Analytic")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    plt.show()

if __name__ in ("__main__"):
    main()
