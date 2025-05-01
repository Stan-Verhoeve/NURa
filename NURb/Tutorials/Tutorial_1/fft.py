import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


def fft(array, N=None):
    array = np.array(array, dtype=np.complex64)
    
    if not N:
        N = array.size

    if N > 2:
        even = deepcopy(array[::2])
        odd = deepcopy(array[1::2])
        
        array[:N//2] = fft(even, N=N//2)
        array[N//2:] = fft(odd, N=N//2)

    for k in range(N//2):
        exp_term = np.exp(2j * np.pi * k / N)
        WH = exp_term * array[k + N//2]
        t = deepcopy(array[k])
        array[k] = t + WH
        array[k + N//2] = t - WH

    return array

def ifft(array, N=None):
    array = np.array(array, dtype=np.complex64)

    if not N:
        N = array.size

    if N > 2:
        even = deepcopy(array[::2])
        odd = deepcopy(array[1::2])
        
        array[:N//2] = ifft(even, N=N//2)
        array[N//2:] = ifft(odd, N=N//2)

    for k in range(N//2):
        exp_term = np.exp(-2j * np.pi * k / N)
        WH = exp_term * array[k + N//2]
        t = deepcopy(array[k])
        array[k] = t + WH
        array[k + N//2] = t - WH

    return array

xx = np.linspace(0, 20, 1024)

testarray = (2 * xx * np.sin(2 * np.pi * xx / 5) + 3 * np.cos(2 * np.pi * xx / 2)) * np.sin(2 * xx)
test = fft(testarray)
recon = ifft(test) / test.size

print(max(recon.imag))
print(min(recon.imag))
plt.figure()
plt.title("Fraction with numpy")
plt.plot(abs(test) / abs(np.fft.fft(testarray)))
plt.savefig("figures/fft.png")

plt.figure()
plt.title("Reconstruction")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(xx, recon.real, c="k", label="Reconstruction")
plt.plot(xx, testarray, ls="--", c="r", label="Original")
plt.savefig("figures/sine.png", bbox_inches="tight", dpi=600)
