import numpy as np

def fft(array):
    array = np.asarray(array, dtype=np.complex64)
    
    def recurse(array):
        N = array.shape[0]
        
        if N <= 1:
            return array

        # Enter recursion on even and odd indices  
        even = recurse(array[::2])
        odd = recurse(array[1::2])
        
        # Compute exponent term for all k
        exp_term = np.exp(-2j * np.pi * np.arange(N // 2) / N).astype(np.complex64)
        # Exponent term is only multiplied by odd indices
        WH = exp_term * odd
        
        # The values 't' are the even indices
        return np.concatenate([even + WH, even - WH])
    
    return recurse(array)

def ifft(array):
    array = np.asarray(array, dtype=np.complex64)
    N = array.shape[0]

    def recurse(array):
        N = array.shape[0]
        if N <= 1:
            return array

        # Enter recursion on even and odd indices
        even = recurse(array[::2])
        odd = recurse(array[1::2])

        # Compute exponent term for all k
        exp_term = np.exp(2j * np.pi * np.arange(N // 2) / N).astype(np.complex64)
        # Exponent term is only multiplied by odd indices
        WH = exp_term * odd

        # The values 't' are the even indices
        return np.concatenate([even + WH, even - WH])
    
    return recurse(array) / N

def fftn(array):
    array = np.asarray(array, dtype=np.complex64)
    ndim = array.ndim

    # Iterate over number of dimensions
    for axis in range(ndim):
        # In principle, FFTN(A) = FFT(FFT(FFT(A, axis=0), axis=1), axis=2, ...)
        # So we are only interested in the data along a single axis at a time
        # Also note that each of the axes in the Fourier transform are independent.
        # This means we can isolate a single axis, and "squash" all the other axes
        # (or dimensions, if you will) into a single 1D slice, so long as we do this
        # consistently and correctly unwrap later. This way, we avoid looping
        # over every other dimension.

        # As such, we wish to reshape our array s.t. we have our axis of interest
        # in one dimension, as the first axis of the reshaped array, and all the
        # other axes squished in the other dimension. We can then iterate over our
        # axis of interest to perform the 1D fourier tranforms on the individual 1D
        # slices of the data

        # As an example in 3D, instead of doing
        # for col in range(array.shape[1]):
        #     for height in range(array.shape[2]):
        #         fft(array[:, col, height])
        #
        # We instead do
        # for col in range(reshaped.shape[1]):
        #     fft(reshaped[:, col])

        # We use np.moveaxis(array, origin, destination) to quickly move the axis
        # of interest to the front, and reshape to be 2D.

        array = np.moveaxis(array, axis, 0)
        shape = array.shape
        array = array.reshape(shape[0], -1)

        for i in range(array.shape[1]):
            array[:,i] = fft(array[:,i])

        array = array.reshape(shape)
        array = np.moveaxis(array, 0, axis)

    return array

def ifftn(array):
    array = np.asarray(array, dtype=np.complex64)
    ndim = array.ndim

    # Iterate over number of dimensions
    for axis in range(ndim):
        # Same logic applies
        array = np.moveaxis(array, axis, 0)
        shape = array.shape
        array = array.reshape(shape[0], -1)

        for i in range(array.shape[1]):
            array[:,i] = ifft(array[:,i])

        array = array.reshape(shape)
        array = np.moveaxis(array, 0, axis)

    return array

def fftfreq(size, d):
    indices = np.arange(size)
    midpoint = size // 2
    indices[indices > midpoint] -= size
    return indices / (d * size)
