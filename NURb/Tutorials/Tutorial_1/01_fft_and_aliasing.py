import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

#########
## Q1a ##
#########

def fft(array, N=None):
    # Cast to complex64
    array = np.array(array, dtype=np.complex64)
    
    # Assume size of array
    if not N:
        N = array.size
    
    # Enter recursion
    if N > 2:
        # Use deepcopy to make sure we're not
        # changing / overwriting pointers later on

        # TODO: This creates 2 new arrays in memory
        #       for each call, so not very memory
        #       efficient. Can we do this without
        #       creating new arrays?
        even = deepcopy(array[::2])
        odd = deepcopy(array[1::2])

        array[: N // 2] = fft(even, N=N // 2)
        array[N // 2 :] = fft(odd, N=N // 2)
    
    for k in range(N // 2):
        # Oscillatory factor
        exp_term = np.exp(2j * np.pi * k / N)

        # Filter function
        WH = exp_term * array[k + N // 2]
        
        # Change array in-place
        t = array[k]
        array[k] = t + WH
        array[k + N // 2] = t - WH

    return array


def ifft(array, N=None):
    # Cast to complex64
    array = np.array(array, dtype=np.complex64)
    
    # Assume size of array
    if not N:
        N = array.size
    
    # Enter recursion
    if N > 2:
        # Use deepcopy to make sure we're not
        # changing / overwriting pointers later on

        # TODO: This creates 2 new arrays in memory
        #       for each call, so not very memory
        #       efficient. Can we do this without
        #       creating new arrays?
        even = deepcopy(array[::2])
        odd = deepcopy(array[1::2])

        array[: N // 2] = ifft(even, N=N // 2)
        array[N // 2 :] = ifft(odd, N=N // 2)
    
    for k in range(N // 2):
        # Oscillatory factor
        exp_term = np.exp(-2j * np.pi * k / N)

        # Filter function
        WH = exp_term * array[k + N // 2]
        
        # Change array in-place
        t = array[k]
        array[k] = t + WH
        array[k + N // 2] = t - WH

    return array

def fftfreq(size, samplerate):
    indices = np.arange(size)
    return (samplerate * (indices - size // 2)) / size

def fftshift(array):
    N = array.size
    return np.roll(array, -N//2)

def function(x):
    # return np.sin(2*np.pi*2*x)
    return (2 * x * np.sin(2 * np.pi * x / 5) + 3 * np.cos(2 * np.pi * x / 2)) * np.sin(2 * x)

def main():
    ###############
    ## Q1b + Q1c ##
    ###############
    Npoints = 8
    xrange = [0,20]
    xx = np.linspace(*xrange, Npoints, endpoint=False)
    xx_plot = np.linspace(*xrange, 2**10, endpoint=False)
    # Function values
    f_of_x = function(xx)
    f_of_x_high = function(xx_plot)
    
    # Fourier transform and reconstruction
    FT = fft(f_of_x)
    freqs = fftfreq(f_of_x.size, 1/np.mean(np.diff(xx)))
    reconstruction = ifft(FT).real / FT.size
    
    # Plotting
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.plot(xx, reconstruction, c="k", label="Reconstruction")
    ax1.plot(xx, f_of_x, c="r", ls="--", label="Original")
    ax1.set(xlabel="x",
            ylabel="y",
            title="Original and reconstructed signal",
            )
    ax1.legend()
    
    ax2.stem(freqs, abs(fftshift(FT)))
    # ax2.plot(freqs, abs(fftshift(FT)))
    ax2.set(xlabel="Freq [Hz]",
            ylabel="|FT|",
            title="Fourier transform of signal",
            xlim=(-2.5,2.5)
            )
    
    fig.tight_layout()
    fig.savefig("figures/Q1.png", bbox_inches="tight", dpi=600)
    
    # Plotting
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.plot(xx_plot, f_of_x_high, c="gray", alpha=0.4, lw=5, label="Original (high sampling rate)")
    ax2.plot(np.fft.fftshift(np.fft.fftfreq(f_of_x_high.size, np.mean(np.diff(xx_plot)))), 
             abs(np.fft.fftshift(np.fft.fft(f_of_x_high))) / f_of_x_high.size, c="gray", alpha=0.4, lw=5, label="Numpy FT")
    for Npoints in [16, 32, 64, 128, 256]:
        xx = np.linspace(*xrange, Npoints, endpoint=False)
            
        # Function values
        f_of_x = function(xx)
        
        # Fourier transform and reconstruction
        FT = fft(f_of_x)
        freqs = fftfreq(f_of_x.size, 1/np.mean(np.diff(xx)))
        reconstruction = ifft(FT).real / FT.size
        
        ax1.plot(xx, reconstruction, label=f"Reconstruction, {Npoints} points")
        
        # ax2.stem(freqs, abs(fftshift(FT)))
        ax2.plot(freqs, abs(fftshift(FT)) / FT.size, label=f"{Npoints} points")
        
    ax1.set(xlabel="x",
            ylabel="y",
            title="Original and reconstructed signal",
            )
    ax1.legend()
    ax2.set(xlabel="Freq [Hz]",
            ylabel="|FT|",
            title="Fourier transform of signal",
            xlim=(-2.5,2.5)
            )
    ax2.legend()
    fig.tight_layout()
    fig.savefig("figures/Q1_many_points.png", bbox_inches="tight", dpi=600)
if __name__ in ("__main__"):
    main()
