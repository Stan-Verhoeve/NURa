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


if __name__ in ("__main__"):
    test_kdtree()
    test_octree()

