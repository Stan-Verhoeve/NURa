import numpy as np
import matplotlib.pyplot as plt
from helperscripts.spatial import octree, KDTree

def get_nodes_at_depth(node, depth):

    if node is None:
        return []
    if node.depth == depth:
        return [node]
    if node.depth > depth:
        return []

    nodes = []
    for child in node.children:
        if child is not None:
            nodes.extend(get_nodes_at_depth(child, depth))

    return nodes

def main():
    # N = np.int64(100_000) 
    N = np.int64(256)**3
    L = np.float32(250.0)
    mp = np.float32(3.64453e10)
    dim = 3
    
    # TODO: Placeholder until data can be loaded
    # pos = np.random.rand(N, dim)
    pos = np.random.normal(size=(N, dim))
    pos += np.abs(np.min(pos, axis=0))
    pos /= np.max(pos, axis=0)
    
    tree = octree(pos, 7)

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
        max_idx = 0
        for node in nodes:
            index = node.index
            if index[0] > max_idx:
                max_idx = index[0]
            
            # TODO: Change back to 0,1,2,3 and multiply by mp
            if index[0] == 2 * (2**level - 1) // 4:
                massmap[0, *index[1:]] = node.length
            
            if index[0] == 1 * (2**level - 1) // 4:
                massmap[1, *index[1:]] = node.length

            if index[0] == 3 * (2**level - 1) // 5:
                massmap[2, *index[1:]] = node.length

            if index[0] == (2**level - 1):
                massmap[3, *index[1:]] = node.length
        
        print("MAX", max_idx)

        x = np.linspace(0, L, pixels)
        y = np.linspace(0, L, pixels)

        fig, ax = plt.subplots(2,2, figsize=(10,8))
        for i in range(4):
            xmin = i * L / 2**level
            xmax = (i + 1) * L / 2**level
            
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
