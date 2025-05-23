import numpy as np
from helperscripts.sorting import merge_sort_indices, merge_sort
import matplotlib.patches as patches

class KDNode:
    def __init__(self, 
                 axis_of_split,
                 coords,
                 depth,
                 left_child,
                 right_child,
                 boundary_coords,
                 start_idx,
                 length):
        
        # Node information
        self.axis_of_split = axis_of_split
        self.coords = coords
        self.depth = depth
        self.left_child = left_child
        self.right_child = right_child
        self.boundary_coords = boundary_coords
        
        # Particle information
        self.start_idx = start_idx
        self.length = length
    
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
    def __init__(self, data, max_depth):
        self.data = data
        self.N, self.dim = self.data.shape
        self.sorted_indices = list(range(self.N))
        self.tree = self._build_tree(0, 0, self.N, 0, max_depth)

    def _build_tree(self, axis, start, length, depth, max_depth, bbox=None):
        
        if depth >= max_depth:
            return None
        if length <= 0:
            return None
        
        # TODO: Should be equivalent. Use this?
        # axis = depth % self.dim
        
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

        
        # Switch to other axis
        # TODO: Needed? See also above
        new_axis = (axis + 1) % self.dim

        # Create children
        left_child = self._build_tree(new_axis, start, median_idx, depth + 1, max_depth, left_bbox)
        right_child = self._build_tree(new_axis, start + median_idx + 1, length - median_idx - 1, depth + 1, max_depth, right_bbox)
         
        # Create node
        node = KDNode(axis, median_coord, depth, left_child, right_child, bbox, start, length)

        return node
   
    def nearest_neighbour(self, point):
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
            """
            # Prune subtrees if bbox is already not a candidate for nearest neighbour
            upper, lower = second.boundary_coords
            # if point outside bbox, closest point is on surface
            # if point inside bbox, closest point is point itself
            # min(upper, point) --> if point above box, move to top edge
            # max(lower, min) --> if point below box, move to bottom edge
            
            # point within bbox: stays the same. Otherwise clipped to edge of bbox
            clipped = np.minimum(upper, np.maximum(point, lower))
            # clipped = np.maximum(lower, np.minimum(point, upper))
            # If distance to bbox worse than current best, we can skip the entire node
            if np.sum((clipped - point)**2) < best[0]:
                print("PRUNED A BRANCH")
                self._nn_search(second, point, best)
            # print("MOVING TO RIGHT")
            """
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


def generate_donut_points(n_points=10_000, r_inner=1.0, r_outer=3.0):
    # Random angles uniformly distributed between 0 and 2π
    theta = np.random.uniform(0, 2 * np.pi, n_points)

    # Radius distributed so area is uniform: r^2 ~ Uniform(r_inner^2, r_outer^2)
    r_squared = np.random.uniform(r_inner**2, r_outer**2, n_points)
    r = np.sqrt(r_squared)

    # Convert to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.vstack((x, y)).T  # Shape (n_points, 2)

def nearest_brute(points):
    distances = np.zeros(points.size)
    
    for i, point1 in enumerate(points):
        best = np.inf
        for j, point2 in enumerate(points):
            if i == j:
                continue
            dist2 = np.sum((point1 - point1)**2)
            if dist2 < best:
                best = dist2
        distances[i] = np.sqrt(best)
    
    return distances

def nearest_tree(points):
    distances = np.zeros(len(points))

    tree = KDTree(points, 50)

    for i, point in enumerate(points):
        nearest = tree.nearest_neighbour(point)
        distances[i] = np.sqrt(nearest[0])
    
    return distances

def nearest_scipy(points):
    from scipy.spatial import KDTree as scipyTree
    distances = np.zeros(len(points))

    scitree = scipyTree(points)
    
    for i, point in enumerate(points):
        nearest = scitree.query(point, 2)[0][1]
        distances[i] = nearest

    return distances

def main():
    import matplotlib.pyplot as plt
    
    # Fix seed for reproducability
    np.random.seed(42)
    n_particles = 10_000
    dim = 2
    positions = np.random.rand(n_particles, dim)
    # positions = np.random.normal(size=(n_particles, dim))
    # positions = generate_donut_points(n_particles)

    # Building tree
    tree = KDTree(positions, 50)

    # Plotting data and tree
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")
    
    ax.scatter(*positions.T, s=1, c="b", alpha=0.3)
    plot_tree(ax, tree.tree, box=True)
    ax.set(xlabel="x",
           ylabel="y",
           )
    fig.savefig("figures/tree", bbox_inches="tight", dpi=600)
    
    # print(tree.tree)
    
    ########################
    ## Nearest neighbours ##
    ########################
    FACTOR = 1
    # Get nearest distances
    nearest_distances_scipy = nearest_scipy(positions) * FACTOR
    nearest_distances_tree = nearest_tree(positions) * FACTOR
    # nearest_distances_brute = nearest_brute(positions) * FACTOR
    
    # Make histograms
    binedges = np.linspace(min(nearest_distances_tree),max(nearest_distances_tree),50)
    hist_scipy, __ = np.histogram(nearest_distances_scipy, bins=binedges)
    hist_tree, __ = np.histogram(nearest_distances_tree, bins=binedges)
    # hist_brute, __ = np.histogram(nearest_distances_brute, bins=binedges)
    
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.stairs(hist_scipy, binedges, ec="gray", lw=4, alpha=0.5, label="Scipy")
    ax.stairs(hist_tree, binedges, ec="k", label="KDTree")
    # ax.stairs(hist_brute, binedges, ec="r", ls="--", label="Brute force")
    ax.set(xlabel="Nearest distance",
           ylabel="Frequency",
           title="Histogram of nearest distances",
           )
    ax.legend()
    fig.savefig("figures/histogram", bbox_inches="tight", dpi=600)
    
    ###############################
    ## Timing of KDTree vs brute ##
    ###############################
    DO_TIMING = False

    if DO_TIMING:
        from timeit import timeit
        Nrepeats = 10
        Npoints = list(range(10,1_000,10))
        brute_timings = np.zeros(len(Npoints))
        tree_timings = np.zeros(len(Npoints))

        for i,N in enumerate(Npoints):
            print(f"Currently working on {N} points")
            positions = np.random.rand(N, dim)
            brute_time = timeit(lambda: nearest_brute(positions), number=Nrepeats) / Nrepeats
            tree_time = timeit(lambda: nearest_tree(positions), number=Nrepeats) / Nrepeats
            
            brute_timings[i] = brute_time
            tree_timings[i] = tree_time

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(Npoints, brute_timings, c="r", marker="^", label="Brute force")
        ax.scatter(Npoints, tree_timings, c="k", marker=".", label="KDTree")
        ax.legend()

        ax.set(xlabel="Number of points",
               ylabel="Evaluation time [s]",
               title="Nearest neighbour search timings",
               )
        fig.savefig("figures/timings", bbox_inches="tight", dpi=600)


if __name__ in ("__main__"):
    main()
