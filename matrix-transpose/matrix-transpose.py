import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A_T = np.array(A)

    if A_T.shape == (2, 2):
        (A_T[0][1], A_T[1][0]) = (A_T[1][0], A_T[0][1])
        return A_T
    else:
        return np.array(list(map(list, zip(*list(A_T)))))