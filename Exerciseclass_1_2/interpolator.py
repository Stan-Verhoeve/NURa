from matplotlib.image import imread

image = imread("M42_128.jpg")


class interpolator:
    def __init__(self, x, y):
        """
        Initialize with known x- and y-values
        """
        self.__x = x
        self.__y = y

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @staticmethod
    def bisection(a, array):
        """
        Find where in array x should be
        using bisection algorithm
        If a is equal to an element in a, will
        return the index of that element
        """
        if a in array:
            return array.index(a)

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

x = [1,2,3,4,5]
y = [2,4,6,8,10]

a = 2
idx = interpolator.bisection(a, x)
print(idx)
print(x)
x.insert(idx, a)
print(x)
