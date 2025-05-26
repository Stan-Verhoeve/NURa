import numpy as np
import matplotlib.pyplot as plt
from helperscripts.spatial import octree, KDTree
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
            if index[0] == 0:
                massmap[0, index[1], index[2]] = node.length * mp
            
            if index[0] == 1:
                massmap[1, index[1], index[2]] = node.length * mp

            if index[0] == 2:
                massmap[2, index[1], index[2]] = node.length * mp

            if index[0] == 3:
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
            
            row, col = divmod(i, 2)
            pcm = ax[row, col].pcolormesh(x, y, massmap[i, :, :])
            ax[row, col].set_aspect("equal", "box")
            fig.colorbar(pcm, ax=ax[row, col], label="Total mass inside node")

            if row == 1:
                ax[row, col].set(xlabel="y [Mpc]")
            if col == 0:
                ax[row, col].set(ylabel="z [Mpc]")

            ax[row, col].set_title(f"Mass distribution in x-range [{xmin:.2f}, {xmax:.2f}] Mpc")
        
        plt.tight_layout()
        plt.savefig(f"figures/Q2b_level{level}.png",dpi=300)
        plt.close()

if __name__ in ("__main__"):
    main()
