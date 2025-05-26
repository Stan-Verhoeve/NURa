import numpy as np
from copy import deepcopy

def fft(array, N=None):
    array = np.asarray(array, dtype=np.complex64)

    if not N:
        N = array.size

    if N > 2:
        even = deepcopy(array[::2])
        odd = deepcopy(array[1::2])

        array[: N // 2] = fft(even, N=N // 2)
        array[N // 2 :] = fft(odd, N=N // 2)

    for k in range(N // 2):
        exp_term = np.exp(2j * np.pi * k / N)
        WH = exp_term * array[k + N // 2]
        t = deepcopy(array[k])
        array[k] = t + WH
        array[k + N // 2] = t - WH

    return array

def fftn(array):
    array = np.asarray(array, dtype=np.complex64)
    ndim = array.ndim
    
    # Iterate over number of dimensions
    for axis in range(ndim):
        # In principle, FFTN(A) = FFT(FFT(FFT(A, axis=0), axis=1), axis=2, ...)
        # So we are only interested in the data along a single axis at a time
        # As such, we wish to reshape our array s.t. we have our axis of interest
        # in one dimension, and all the rest in the other dimension. We can then 
        # iterate over our axis of interest to perform the 1D fourier tranforms
        # on the individual 1D slices of the data

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

def ifft(array, N=None):
    array = np.asarray(array, dtype=np.complex64)
    if not N:
        N = array.size
    
    def recurse(array, N):
        if N > 2:
            even = deepcopy(array[::2])
            odd = deepcopy(array[1::2])

            array[: N // 2] = recurse(even, N=N // 2)
            array[N // 2 :] = recurse(odd, N=N // 2)

        for k in range(N // 2):
            exp_term = np.exp(-2j * np.pi * k / N)
            WH = exp_term * array[k + N // 2]
            t = deepcopy(array[k])
            array[k] = t + WH
            array[k + N // 2] = t - WH

        return array

    array = recurse(array, N)
    return array / N

def fftfreq(size, samplerate):
    indices = np.arange(size)
    return (samplerate * (indices - size // 2)) / size
