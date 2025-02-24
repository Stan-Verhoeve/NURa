import numpy as np

# TODO: make class for matrix??
def to_lu(matrix: np.ndarray) -> np.ndarray:
    # TODO:
    # Decomposes in U,V (returns U,V). Better
    # to overwrite in-place. To do later

    # Check if matrix is square
    if len(set(np.shape(matrix))) > 1:
        raise ValueError("Matrix should be square for this algorithm")
    
    # Explicitly cast to float64 ndarray
    matrix = np.array(matrix, dtype=np.float64)
    
    # In-place matrix
    inplace_matrix = matrix.copy()
    N = len(matrix)
    
    # Indexing array (keeps track of permutations of row swaps)
    permutation = np.arange(N)
    
    # Implicit pivot; largest coef on each row
    largest_coef = np.max(np.abs(matrix), axis=-1)
    if np.any(largest_coef == 0):
        raise ValueError("Matrix is singular")

    largest_coef_inv = largest_coef ** (-1)

    # Loop over columns
    for k in range(N):
        # Find index of largest pivot candidate
        imax = k + np.argmax(np.abs(matrix[k:,k] * largest_coef_inv[permutation][k:]))
        
        # If not on the diagonal, swap rows
        if imax != k:
            # swap rows
            inplace_matrix[[imax, k], :] = inplace_matrix[[k, imax], :]
            # Track permutation
            permutation[k], permutation[imax] = permutation[imax], permutation[k]
        
        for i in range(k+1, N):
            inplace_matrix[i, k] /= inplace_matrix[k, k]
            # Loop over j is identical to dot product
            inplace_matrix[i, k+1:] -= np.dot(inplace_matrix[i, k], inplace_matrix[k, k+1:])

    # idx_array essentially stores the permutation destinations (i.e. idx0 should go to the
    # idx stored in the array, etc). For numpy indexing, however, we want the inverse permutation
    # that has in idx0 the row that should go to 0, not that comes from 0. As such, invert the
    # permutation
    inv_permutation = np.empty_like(permutation)
    inv_permutation[permutation] = np.arange(N)


    return inplace_matrix, inv_permutation

# TODO: make single function?
def forward_substition(L, y):
    N = len(y)
    z = np.zeros(N)

    for i in range(N):
        z[i] = y[i] - np.dot(L[i, :i], z[:i])
    return z

# TODO: make single function?
def backward_substition(U, y):
    N = len(y)
    z = np.zeros(N)

    for i in range(N)[::-1]:
        z[i] = (y[i] - np.dot(U[i, i:], z[i:])) / U[i,i]
    return z


# TODO: move to helper script?
def polynomial(coefs, x):
    y = 0.
    for power,c in enumerate(coefs):
        y += c * x**power

    return y

    
def main():
    from helper_scripts.interpolator import interpolator
    from helper_scripts.pretty_printing import pretty_print_array
    import matplotlib.pyplot as plt
    
    # Get x, y data
    x,y = np.genfromtxt("Vandermonde.txt").T
    
    # Construct Vandermonde matrix
    V = x[:,None] ** np.arange(len(x))
    
    LU, permutation = to_lu(V)
    U = np.triu(LU)
    L = np.tril(LU, k=-1) + np.eye(*LU.shape)
    
    # Sanity check
    reconstruction = (L@U)[permutation, :]
    print("Sanity check")
    print(f"LU = V: {np.all(np.isclose(V, reconstruction))}\n")

    # TODO: clean up code
    # TODO: do I want classes / more functions?

    # Intermediate solution vector
    z = forward_substition(L, y[permutation])

    # Coefficient vector
    c = backward_substition(U, z)

    # Pretty print coefficients
    print("Polynomial coefficients:")
    pretty_print_array(c, ncols=4)
    

    # TODO: Clean this up?
    # TODO: Also do for 1 iteration
    # Improve using iterative approach
    c_iter = c.copy()
    Niters = 10
    for __ in range(Niters):
        dy = V @ c_iter - y

        # Intermediate
        z = forward_substition(L, dy[permutation])
        # Error in c
        dc = backward_substition(U, z)
        # Subtract to minimise
        c_iter -= dc
    
    # Create interpolator object
    ipl = interpolator(x, y)

    # Points to interpolate on
    interp_x = np.linspace(min(x), max(x), 1000)
    
    # Interpolate using Neville
    neville_y = ipl.interpolate(interp_x, kind="neville")

    # Get polynomial using Vandermonde coefficients
    LU_y = polynomial(c, interp_x)
    LU_y_iter = polynomial(c_iter, interp_x)
    
    # Get absolute differences
    abs_diff_neville = abs(y - ipl.interpolate(x, kind="neville"))
    abs_diff_LU = abs(y - polynomial(c, x))
    abs_diff_LU_iter = abs(y - polynomial(c_iter, x))

    # TODO: Create figure for each subquestion (a), (b), (c)?
    # TODO: Write function to plot for me?
    # Create figure
    aspect = 2
    fig, (ax1, ax2) = plt.subplots(2,1, 
                                   gridspec_kw={"height_ratios":[aspect, 1]}, 
                                   figsize=(8,6), 
                                   sharex=True,
                                   constrained_layout=True,
                                   dpi=300)  
    
    ax1.set_ylim(-400, 400)
    ax1.set_ylabel("y")
    ax1.scatter(x,y, label="Nodes")
    ax1.plot(interp_x, neville_y, c="green", ls="--", label="Neville")
    ax1.plot(interp_x, LU_y, c="orange", label="LU decomposition")
    ax1.plot(interp_x, LU_y_iter, c="purple", ls="-.", label=f"LU, {Niters} iterations")
    
    ax2.set_xlabel("x")
    ax2.set_ylabel("|$y_i - y$|")
    ax2.plot(x, abs_diff_neville, c="green", ls="--")
    ax2.plot(x, abs_diff_LU, c="orange")
    ax2.plot(x, abs_diff_LU_iter, c="purple")
    ax2.set_yscale("log")

    ax1.legend()
    
    plt.savefig("figures/02_vandermonde.png", bbox_inches="tight", dpi=300)
    
    # TODO: use timeit to time different approaches

if __name__ in ("__main__"):
    main()
