import numpy as np

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
    new_matrix = np.zeros_like(matrix)    
    N = len(matrix)
    
    # Identity and zero matrix
    # TODO: remove once in-place is implemented
    L = np.eye(N, dtype=np.float64)
    U = np.zeros_like(matrix)
    
    # Loop over columns
    for j in range(N):
        for i in range(j+1):
            # sum(L[i,k] * U[k,j]) is simply the dot product between L_ik and U_kj
            # Since k goes from 0 to i-1, this is L[i, :i]
            U[i, j] = matrix[i, j] - np.dot(L[i, :i], U[:i, j])
            new_matrix[i,j] = matrix[i, j] - np.dot(L[i, :i], U[:i, j])

        for i in range(j+1, N):
            # sum(L[i,k] * U[k,j]) is simply the dot product between L_ik and U_kj
            # Since k goes form 0 to j-1, this is L[i, :j]
            L[i, j] = matrix[i, j] - np.dot(L[i, :j], U[:j, j])
            new_matrix[i,j] = matrix[i, j] - np.dot(L[i, :j], U[:j, j])
        
        # Division by beta_jj
        L[j+1:,j] /= U[j,j]
        new_matrix[j+1:,j] /= U[j,j]

    return L, U, new_matrix

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

# TODO: move to helper script?
def pretty_print_array(array, formatter=".2e", ncols=4):
    N = len(np.asarray(array).flatten())
    nrows = int(np.ceil(N / ncols))

    padded = np.pad(array, (0, nrows * ncols - N), constant_values = np.nan)
    padded = padded.reshape(nrows,ncols)

    format_func = lambda x: f"{x:{formatter}}" if not np.isnan(x) else "     "

    print(np.array2string(padded, formatter={"float_kind":format_func}, separator="   "))

    return


    
def main():
    from helper_scripts.interpolator import interpolator
    import matplotlib.pyplot as plt
    
    # Get x, y data
    x,y = np.genfromtxt("Vandermonde.txt").T
    
    # Construct Vandermonde matrix
    V = x[:,None] ** np.arange(len(x))
    
    L, U, B = to_lu(V)
    
    # Sanity check
    print("Sanity check")
    print(f"LU = V: {np.all(np.isclose(V, L@U))}\n")

    # TODO: clean up code
    # TODO: do I want classes / more functions?

    # Intermediate solution vector
    z = forward_substition(L, y)

    # Coefficient vector
    c = backward_substition(U, z)

    # Pretty print coefficients
    print("Polynomial coefficients:")
    pretty_print_array(c, ncols=4)
    

    # TODO: Make this cleaner
    # Improve using iterative approach
    c_iter = c.copy()
    Niters = 10
    for __ in range(Niters):
        dy = V @ c_iter - y

        # Intermediate
        z = forward_substition(L, dy)
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

    # TODO: find nicer colours / linestyles
    # Create figure
    aspect = 2
    fig, (ax1, ax2) = plt.subplots(2,1, 
                                   gridspec_kw={"height_ratios":[aspect, 1]}, 
                                   figsize=(8,6), 
                                   sharex=True,
                                   constrained_layout=True,
                                   dpi=300)  
    # Create figure
    ax1.set_ylabel("y")
    ax1.scatter(x,y, label="Nodes")
    ax1.plot(interp_x, neville_y, c="k", label="Neville")
    ax1.plot(interp_x, LU_y, c="b", ls="--", label="LU decomposition")
    ax1.plot(interp_x, LU_y_iter, c="orange", ls="-.", label=f"LU, {Niters} iterations")
    
    ax2.set_xlabel("x")
    ax2.set_ylabel("|$y_i - y$|")
    ax2.plot(x, abs_diff_neville, c="k")
    ax2.plot(x, abs_diff_LU, c="b")
    ax2.plot(x, abs_diff_LU_iter, c="orange")
    ax2.set_yscale("log")

    ax1.legend()
    
    plt.savefig("figures/02_vandermonde.png", bbox_inches="tight", dpi=300)
    
    # TODO: use timeit to time different approaches

if __name__ in ("__main__"):
    main()
