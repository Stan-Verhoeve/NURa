import numpy as np
from helperscripts.sorting import merge_sort_indices
from dataclasses import dataclass

@dataclass
class KDNode:
    # Node information
    axis_of_split: int
    coords: tuple[float, ...]
    depth: int
    left_child: "KDNode"
    right_child: "KDNode"
    boundary_coords: tuple[tuple[float,...],...]

    # Data information
    start_idx: int
    length: int

class KDTree:
    """
    Class that builds and handles KDTree, and includes NN search


    """
    def __init__(self, data, max_depth):
        self.data = data
        self.N, self.dim = self.data.shape
        self.sorted_indices = list(range(self.N))
        self.max_depth = max_depth

        self.tree = self._build_tree(0, self.N, 0)
        
    def _build_tree(self, start, length, depth, bbox=None):
        # Return if at max depth, or if no points in child
        if depth > self.max_depth:
            return None
        if length <= 0:
            return None
         
        # Change axis
        axis = depth % self.dim

        # Indices of current node
        indices = self.sorted_indices[start:start+length]

        # Get data and indices sorted by data
        data_1d = self.data[:,axis]
        new_sorted = np.array(merge_sort_indices(indices, data_1d))

        # Position of index for median
        median_idx = length // 2
        # Index of median value
        median_val_idx = new_sorted[median_idx]
        median_coord = self.data[median_val_idx]

        # Update index array
        self.sorted_indices[start:start + length] = new_sorted

        # Construct bbox based on data of current node
        if bbox is None:
            lower_left = np.min(self.data[indices], axis=0)
            upper_right = np.max(self.data[indices], axis=0)
            bbox = (lower_left, upper_right)

        # Split bbox for each child
        lower, upper = bbox
        left_upper = upper.copy()
        right_lower = lower.copy()

        # Upper point of left child should be median
        # Lower point of right child should be median
        left_upper[axis] = median_coord[axis]
        right_lower[axis] = median_coord[axis]

        left_bbox = (lower, left_upper)
        right_bbox = (right_lower, upper)

        # Create children
        left_child = self._build_tree(start, median_idx, depth + 1, left_bbox)
        right_child = self._build_tree(start + median_idx + 1, length - median_idx - 1, depth + 1, right_bbox)

        # Create node
        node = KDNode(axis, median_coord, depth, left_child, right_child, bbox, start, length)

        return node

    def nearest_neighbour(self, point):
        """
        Performs nearest neighbour search in the current tree

        Parameters
        ----------
        point : array-like
            Point to find nearest neighbour to

        Returns
        -------

        """
        self.total_checks = 0
        best = [np.inf, None]
        self._nn_search(self.tree, point, best)
        # print(self.total_checks)
        # print("BEST", best)
        return best[0], best[1]

    def _nn_search(self, node, point, best):
        if node is None:
            return

        # Find in which node the point falls
        axis = node.axis_of_split
        curr_val = point[axis]
        go_left = curr_val < node.coords[axis]

        # Check in this order
        first = node.left_child if go_left else node.right_child
        second = node.right_child if go_left else node.left_child

        # Brute-force leaf nodes
        if first is None and second is None:
            start = node.start_idx
            end = start + node.length
            for i in self.sorted_indices[start:end]:
                self.total_checks += 1
                dist2 = np.sum((self.data[i] - point)**2)
                if 0 < dist2 < best[0]:
                    # print("CURRENT BEST", dist2)
                    best[0] = dist2
                    best[1] = i

        # Check current point (coordinate of split)
        dist2 = np.sum((node.coords - point)**2)
        if 0 < dist2 < best[0]:
            best[0] = dist2
            median_idx = node.start_idx + node.length//2
            best[1] = self.sorted_indices[median_idx]

        # Recurse
        self._nn_search(first, point, best)

        # Check if we need to visit other child
        if (node.coords[axis] - point[axis])**2 < best[0]:
            self._nn_search(second, point, best)

@dataclass
class octnode:
    depth: int
    index : tuple[int, int, int] # Position in space; grid index
    pos: list[float, float, float] # Position in space; midpoint of node
    children: tuple["octnode",...]  # list of 8 children

    # Child logic:
    # Let us label the children 1-8, and denote `-` when their center is
    # lower than that of its parent, and `+` when their center is higher 
    # than that of its parent
    #     1  2  3  4  5  6  7  8
    # X : -  -  -  -  +  +  +  +
    # Y : -  -  +  +  -  -  +  +
    # Z : -  +  -  +  -  +  -  +

    # So the recursion keeps going down to lower-left-back for its first child,
    # then move up one for second child, move one right (and down) for third, 
    # move one up for fourth, etc.
    
    # Particle info
    start_idx : int
    length: int

class octree:
    def __init__(self, data, max_depth):
        self.data = data
        self.N, self.dim = self.data.shape
        if self.dim != 3:
            raise ValueError(f"Expected number of dimensions to be 3, not {self.dim}")
        
        # Node logic
        self.sorted_indices = list(range(self.N))
        self.max_depth = max_depth
        # Offset direction for each child
        # Amounts to binary counting to 7
        self.offsets = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],])
        
        # Assume normalized coordinates (full volume extends from (0,0,0) to (1,1,1))
        index = np.array([0,0,0])
        center = np.array([0.5, 0.5, 0.5])
        size = 1.0

        self.tree = self._build_tree(index, center, size, 0, 0, self.N)
        
    def _build_tree(self, index, center, size, depth, start, length):
        if depth > self.max_depth:
            return
        if length <= 0:
           return
        
        # Midpoint of the current node
        mid = size / 2

        # Center coordinates
        cx, cy, cz = center

        # Split particles into their corresponding octants
        particle_indices = self.sorted_indices[start:start+length]
        points = self.data[particle_indices]
        oct_idx = [[] for _ in range(8)]
        
        # Look up in which octant particle falls
        # Use `-` and `+` logic from before
        
        # Child  : 1  2  3  4  5  6  7  8
        # X      : -  -  -  -  +  +  +  +
        # Y      : -  -  +  +  -  -  +  +
        # Z      : -  +  -  +  -  +  -  +
        # Octant : 0  1  2  3  4  5  6  7

        # We check if coordinate is larger than center
        # This gives a boolean table that we need to map
        # to octants as described above. In essence, we need
        # True, True, True --> 7
        # False, False, False --> 0
        # Everything else in between
        # This is just binary counting to 7, where the 
        # x-coordinate corresponds to the 4 bit, 
        # y-coordinate corresponds to the 2 bit,
        # z-coordinate corresponds to the 1 bit
        # As such, we bit-shift the booleans by 2, 1, 0 respectively
        # and take logical or to get final binary number
        bools = points >= center
        octants = (bools[:,0].astype(int) << 2) | \
                  (bools[:,1].astype(int) << 1) | \
                  (bools[:,2].astype(int) << 0)

        for i in range(length):
            # Add particle indices for this child
            octant = octants[i]
            oct_idx[octant].append(particle_indices[i])

        children = np.array([None] * 8)
        cursor = start
        for i, offset in enumerate(self.offsets):
            # Number of particles in this child
            num = len(oct_idx[i])

            # Update sorted_indices 
            self.sorted_indices[cursor:cursor+num] = oct_idx[i]
            
            # Child index
            child_index = 2 * index + offset
            # Center point in space coordinates
            child_center = center + (offset - 0.5) * mid
            child = self._build_tree(child_index, child_center, mid, depth + 1, cursor, num)
            children[i] = child
            
            # Increment start position
            cursor += num

        return octnode(depth, index, center, children, start, length)


def plot_2Dtree(tree, ax=None, color="k", linedwidth=0.5, box=False):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if ax is None:
        fig = plt.figure()
        fig.add_subplot(111)
    
    def recurse(node):
        if node is None:
            return

        axis = node.axis_of_split
        point = node.coords
        lower_left, upper_right = node.boundary_coords
        
        xmin, ymin = lower_left
        xmax, ymax = upper_right
        
        if box:
            # Width and height of box
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height,
                                     linewidth=1, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
        else:
            if axis == 0:
                ax.plot([point, point], [ymin, ymax], c=color, lw=linewidth)
            else:
                ax.plot([xmin, xmax], [point, point], c=color, lw=linewidth)
        
        recurse(node.left_child)
        recurse(node.right_child)

    recurse(tree)

    return

def plot_octree(tree, ax=None, initial_size=1.0, color="k", linewidth=0.5, only_leaves=False):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    # Keep track of line objects for cubes
    lines = []

    def cube_edges(center, size):
        """Edges of a cube"""
        c = np.array(center)
        h = size / 2
        # Corner coordinates
        corners = np.array([[x, y, z] for x in [-h, h] for y in [-h, h] for z in [-h, h]])
        # Translate to center
        corners += c
    
        # Edges defined by their corner indices
        # Here, (0,1) means the edge going from corner 0 to corner 1, etc.
        edges = [
            (0,1), (0,2), (0,4),
            (1,3), (1,5),
            (2,3), (2,6),
            (3,7),
            (4,5), (4,6),
            (5,7),
            (6,7)
        ]
        return [(corners[i], corners[j]) for i, j in edges]

    def recurse(node, size):
        if node is None:
            return
        # If all children are None (so no children), we have a leaf
        is_leaf = all(child is None for child in node.children)
        # Only add cube if we need to
        if not only_leaves or is_leaf:
            lines.extend(cube_edges(node.pos, size))
        
        child_size = size / 2
        for child in node.children:
            recurse(child, child_size)

    recurse(tree, initial_size)

    # Add lines to a line collection...
    lc = Line3DCollection(lines, colors=color, linewidths=linewidth)
    # ...and plot the collection
    ax.add_collection3d(lc)
    
    ax.set_box_aspect([1, 1, 1])
    # ax.view_init(elev=30, azim=30)

