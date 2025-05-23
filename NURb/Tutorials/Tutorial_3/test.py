import numpy as np

class KDNode:
    def __init__(self, point, left=None, right=None, axis=0):
        self.point = point
        self.left = left
        self.right = right
        self.axis = axis

def build_kd_tree(points, depth=0):
    if len(points) == 0:
        return None

    k = points.shape[1]  # number of dimensions
    axis = depth % k     # cycle through axes

    # sort points along the selected axis
    sorted_idx = points[:, axis].argsort()
    sorted_points = points[sorted_idx]

    median_idx = len(sorted_points) // 2
    median_point = sorted_points[median_idx]

    # recursively build subtrees
    left_subtree = build_kd_tree(sorted_points[:median_idx], depth + 1)
    right_subtree = build_kd_tree(sorted_points[median_idx + 1:], depth + 1)

    return KDNode(median_point, left_subtree, right_subtree, axis)

# Example usage
data = np.random.rand(100, 2)  # 10 points in 2D
data = np.random.normal(size=(10000, 2))
kd_tree_root = build_kd_tree(data)

import matplotlib.pyplot as plt

def plot_kd_tree(node, bounds, depth=0):
    if node is None:
        return

    axis = node.axis
    x, y = node.point

    # Draw the splitting line
    if axis == 0:
        # Vertical line
        plt.plot([x, x], [bounds[1][0], bounds[1][1]], 'r-')
        left_bounds = ([bounds[0][0], x], bounds[1])
        right_bounds = ([x, bounds[0][1]], bounds[1])
    else:
        # Horizontal line
        plt.plot([bounds[0][0], bounds[0][1]], [y, y], 'b-')
        left_bounds = (bounds[0], [bounds[1][0], y])
        right_bounds = (bounds[0], [y, bounds[1][1]])

    # Recurse
    plot_kd_tree(node.left, left_bounds, depth + 1)
    plot_kd_tree(node.right, right_bounds, depth + 1)

def show_kd_tree(root, points):
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c='black')
    bounds = [[min(points[:, 0]), max(points[:, 0])],
              [min(points[:, 1]), max(points[:, 1])]]
    plot_kd_tree(root, bounds)
    plt.title("KD-Tree Partitioning")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.savefig("figures/chat", bbox_inches="tight", dpi=600)

show_kd_tree(kd_tree_root, data)
