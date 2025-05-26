import numpy as np
import matplotlib.pyplot as plt

def test_kdtree():
    from helperscripts.spatial import KDTree, plot_2Dtree
    N = 10_000
    dim = 2
    data = np.random.rand(N, dim)

    tree = KDTree(data, 7)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")

    ax.scatter(*data.T, alpha=0.3, s=2)
    plot_2Dtree(tree.tree, ax, box=True)

    ax.set(xlabel="x",
           ylabel="y",
           title="KDTree",
           )
    fig.savefig("figures/test_spatial", bbox_inches="tight", dpi=600)

def test_octree():
    from helperscripts.spatial import octree, plot_octree
    # import matplotlib
    # matplotlib.use("TkAgg")
    np.random.seed(42)
    N = 10000
    dim = 3
    data = np.random.rand(N, dim)
    
    print("Now building octree")
    tree = octree(data, 3)
    # print(tree.tree)
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*data.T, s=1, alpha=0.3)
    print("Now plotting octree")
    plot_octree(tree.tree, ax, only_leaves=True)
    fig.savefig(f"figures/test_octree", bbox_inches="tight", dpi=600)

def test_fft():
    from helperscripts.fft import fft, ifft
    
    freq = 5
    time = np.linspace(0, 1, 1024, endpoint=False)
    testdata = np.sin(2 * np.pi * time * freq)

    freqs = np.fft.fftfreq(testdata.size, np.mean(np.diff(time)))
    npFT = np.fft.fft(testdata)
    FT = fft(testdata)
    
    npiFT = np.fft.ifft(npFT)
    iFT = ifft(FT.copy())
    
    fig = plt.figure(figsize=(9,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.plot(np.fft.fftshift(freqs), np.fft.fftshift(np.abs(FT)), label="Own FFT")
    ax1.plot(np.fft.fftshift(freqs), np.fft.fftshift(np.abs(npFT)), c="gray", lw=5, alpha=0.3, label="np.fft.fft")
    
    ax1.set(xlabel="Frequency [Hz]",
            ylabel="Magnitude",
            xlim=(-1.5*freq, 1.5*freq))
    
    ax2.plot(time, iFT.real, label="Own iFFT")
    ax2.plot(time, npiFT.real, c="gray", lw=5, alpha=0.3, label="np.fft.ifft")

    ax2.set(xlabel="Time [s]",
            ylabel="Amplitude",
            )
    
    fig.tight_layout()
    fig.savefig("figures/test_fft", bbox_inches="tight", dpi=600)

def test_fftn():
    from helperscripts.fft import fftn, ifftn
    x = np.linspace(-1, 1, 16)
    y = np.linspace(-1, 1, 16)
    # z = np.linspace(-1, 1, 64)
    xx, yy = np.meshgrid(x, y, sparse=True)
    sigma = 0.1
    testdata = 1/(np.sqrt(2*np.pi * sigma**2)) * np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    
    npfft = np.fft.fftn(testdata)
    fft = fftn(testdata)
    
    npifft = np.fft.ifftn(npfft)
    ifft = ifftn(fft.copy())
    
    tolerance = 1e-6
    fft_close_to_np = np.all(np.isclose(np.abs(npfft), np.abs(fft), atol=tolerance))
    ifft_close_to_np = np.all(np.isclose(np.abs(npifft), np.abs(ifft), atol=tolerance))
    
    print("FFT close to numpy:", fft_close_to_np)
    print("iFFT close to numpy:", ifft_close_to_np)
if __name__ in ("__main__"):
    test_kdtree()
    test_octree()
    test_fft()
    test_fftn()

