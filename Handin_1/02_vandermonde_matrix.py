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
    new_matrix = np.zeros_like(matrix)    
    N = len(matrix)
    
    # Identity and zero matrix
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

A = np.array([[2,3,1],
              [4,7,3],
              [6,18,5]], dtype=np.float64)

x,y = np.genfromtxt("Vandermonde.txt").T

vdm = x[:,None] ** np.arange(len(x))

L, U, B = to_lu(vdm)

print(f"LU = A: {np.all(np.isclose(vdm, L@U))}")


