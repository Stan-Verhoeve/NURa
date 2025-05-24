import numpy as np
from helperscripts.sorting import merge_sort_indices
import matplotlib.patches as patches
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

    def __repr__(self):
        def recurse(node, prefix="", is_left=True):
            if node is None:
                return prefix + ("└── " if is_left else "├── ") + "None\n"

            node_str = prefix + ("└── " if is_left else "├── ") + f"depth: {node.depth}\n"
            new_prefix = prefix + ("    " if is_left else "│   ")
            node_str += new_prefix + f"aos: {node.axis_of_split}\n"
            node_str += new_prefix + f"coords: {node.coords}\n"
            node_str += new_prefix + f"bbox: {node.boundary_coords}\n"

            node_str += new_prefix + "left child:\n"
            node_str += recurse(node.left_child, new_prefix + "    ", True)
            node_str += new_prefix + "right child:\n"
            node_str += recurse(node.right_child, new_prefix + "    ", False)

            return node_str

        return "KDTree\n" + recurse(self, "", True)


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

def plot_tree(ax, node, box=False):
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
                                 linewidth=1, edgecolor="k", facecolor="none", alpha=0.3)
        ax.add_patch(rect)
    else:
        if axis == 0:
            ax.plot([point, point], [ymin, ymax], c="k", lw=1.2, alpha=0.3)
        else:
            ax.plot([xmin, xmax], [point, point], c="k", lw=1.2, alpha=0.3)
    
    plot_tree(ax, node.left_child, box)
    plot_tree(ax, node.right_child, box)

    return

@dataclass
class octnode:
    depth: int
    pos: list[int,...]
    Mtot: float
    com: float
    children: tuple["octnode",...]


class octree:
    def __init__(self, data, max_depth):
        self.data = data
        self.N, self.dim = self.data.shape
        if self.dim != 3:
            raise ValueError(f"Expected number of dimensions to be 3, not {self.dim}")

        self.sorted_indices = list(range(self.N))
        self.max_depth = max_depth

        self.tree = self._build_tree(0, self.N, 0)

    def _build_tree(self, start, length, depth, bbox=None):
        if depth > self.max_depth:
            return
        if length <= 0:
            return
        
        NotImplemented

