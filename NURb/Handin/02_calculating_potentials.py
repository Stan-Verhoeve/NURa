import numpy as np
import matplotlib.pyplot as plt
from helperscripts.spatial import octree, KDTree
from helperscripts.fft import fftn, ifftn, fftfreq
import h5py

def get_nodes_at_depth(node, depth):
    if node is None:
        # Node is a leaf or empty
        return []
    if node.depth == depth:
        # Node is exactly at depth
        return [node]
    if node.depth > depth:
        # Not interested
        return []

    nodes = []
    # Iterate over node children (i.e. recurse)
    for child in node.children:
        if child is not None:
            # Only extend if child exists
            nodes.extend(get_nodes_at_depth(child, depth))

    return nodes

def main():
    np.random.seed(42)

    # TODO: Temporary to work on my desktop as well, remove later
    try:
        with h5py.File("/data2/daalen/DMO_a0.1_256.hdf5","r") as handle:
            pos = handle["Position"][...] #particle positions, shape (Np,3), comoving
    except:
        with h5py.File("DMO_a0.1_256.hdf5","r") as handle:
            pos = handle["Position"][...] #particle positions, shape (Np,3), comoving
    
    # Box and particle information
    N = np.int64(256)**3               # Number of particles
    L = np.float32(250.0)              # Side length in Mpc
    mp = np.float32(3.64453e10)        # Particle mass in Msun
    
    # Simulation information
    G = np.float32(4.3009e-9)          # Gravitational constant in Mpc*(km/s)^2/Msun
    h = np.float32(0.3755)             # Hubble parameter
    scale_factor = np.float32(0.1)     # Scale factor a
    redshift = 1.0 / scale_factor - 1  # Redshift z
    rho_mean = N * mp / L**3           # Mean density in Msun/Mpc^3
    
    # Scale positions to be in range (0,1)^3
    # TODO: Change tree to work on non-normalized data?
    pos /= L

    # pos = np.random.normal(size=(N, 3))
    # pos = np.random.rand(N, 3)
    # pos += np.abs(np.min(pos, axis=0))
    # pos /= np.max(pos, axis=0)
    
    ########################
    ## Q2a, building tree ##
    ########################
    tree = octree(pos, 7)
    
    # Iterate over the levels
    for level in [3, 5, 7]:
        pixels = 2**level
        massmap=np.zeros((4, pixels, pixels), dtype=np.float32)

        nodes = get_nodes_at_depth(tree.tree, level)
        
        # TODO: Currently "iterates" over tree twice
        #       First to grab the nodes at level
        #       Second time iterate over all these nodes
        #       to find those at specific index
        #       Is it possible bo to direct index lookup?
        #       If so, we should probably use that instead
        # Iterate over nodes to get indexing info
        for node in nodes:
            index = node.index
            
            # Extract index; TODO can we do this more elegantly?
            if index[0] == 0: # * (2**level - 1) // 4:
                massmap[0, index[1], index[2]] = node.length * mp
            
            if index[0] == 1: # * (2**level - 1) // 4:
                massmap[1, index[1], index[2]] = node.length * mp

            if index[0] == 2: # * (2**level - 1) // 4:
                massmap[2, index[1], index[2]] = node.length * mp

            if index[0] == 3: # * (2**level - 1) // 4:
                massmap[3, index[1], index[2]] = node.length * mp
        
        
        # TODO: for plotting on (0,L). Change to pixel instead?
        x = np.linspace(0, L, pixels)
        y = np.linspace(0, L, pixels)
        
        # Create figure and plot
        fig, ax = plt.subplots(2,2, figsize=(10,8))
        for i in range(4):
            # Get current x-range
            # At this level, we have split 2**level times, meaning each slice
            # has width L / (2**level)
            xmin = i * L / (2**level)
            xmax = (i + 1) * L / (2**level)
            # xmin = (i * L * (2**level - 1) // 4) / (2**level)
            # xmax = xmin + 1/(2**level)

            row, col = divmod(i, 2)
            pcm = ax[row, col].pcolormesh(x, y, massmap[i, :, :], shading="auto")
            ax[row, col].set_aspect("equal", "box")
            fig.colorbar(pcm, ax=ax[row, col], label="Total mass inside node")

            if row == 1:
                ax[row, col].set(xlabel="y [Mpc]")
            if col == 0:
                ax[row, col].set(ylabel="z [Mpc]")

            ax[row, col].set_title(f"Mass distribution x $\in$ [{xmin:.2f}, {xmax:.2f}] Mpc")
        
        plt.tight_layout()
        plt.savefig(f"figures/Q2a_level{level}.png",dpi=300)
        plt.close()
    
    
    ##################
    ## Q2b, fourier ##
    ##################
    print("STARTING FOURIER")
    density = np.zeros((128, 128, 128))
    leaves = get_nodes_at_depth(tree.tree, 7)
    leaf_volume = (L / 128) ** 3

    # k-vector in each direction is identical
    dk = 2 * np.pi / L
    # TODO: Currently uses fftfreq --> change to own!!
    k_1d = dk * np.fft.fftfreq(128, 128 / L)
    kx, ky, kz = np.meshgrid(k_1d, k_1d, k_1d, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2

    # zero component is mean density, so set k2 to inf to
    # ensure Fourier becomes zero there
    k2[0, 0, 0] = np.inf

    # Populate the mass matrix
    for leaf in leaves:
        index = leaf.index
        density[index[0], index[1], index[2]] = leaf.length * mp
    
    density /= leaf_volume
    
    phi_hat = fftn(density).copy() / k2
    potential = -G * np.abs(ifftn(phi_hat)) / np.pi
    potential[0, 0, 0] = 0.
    
    # For plotting extent
    x = np.linspace(0, L, 128)
    y = np.linspace(0, L, 128)

    # Create figure and plot
    fig, ax = plt.subplots(2,2, figsize=(10,8))
    slices = [0, 16, 32, 64]
    for i in range(4):
        row, col = divmod(i, 2)
        pcm = ax[row, col].pcolormesh(x, y, potential[slices[i], :, :], shading="auto")
        ax[row, col].set_aspect("equal", "box")
        fig.colorbar(pcm, ax=ax[row, col], label="Potential inside node")

        if row == 1:
            ax[row, col].set(xlabel="y [Mpc]")
        if col == 0:
            ax[row, col].set(ylabel="z [Mpc]")

        ax[row, col].set_title(f"Potential of slice $x_{{{slices[i]}}}$")
    
    plt.tight_layout()
    plt.savefig(f"figures/Q2b_potential.png",dpi=300)
    plt.close()
if __name__ in ("__main__"):
    main()
