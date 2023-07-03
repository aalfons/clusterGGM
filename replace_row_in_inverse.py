import numpy as np


def updateInverse(inverse, vec, i):
    Au = inverse[:, i]
    vA = vec.T @ inverse
    N = np.outer(Au, vA)
    D = 1 / (1 + vA[i])

    result = inverse - D * N

    Au = result @ vec
    vA = result[i, :]
    N = np.outer(Au, vA)
    D = 1 / (1 + Au[i])

    result -= D * N

    return result


def dropVariable(A, i):
    result = np.delete(A, i, axis=0)
    result = np.delete(result, i, axis=1)

    return result


def removeAndInsertInplace(mat, r, i):
    if i > r:
        # Store the row/column that will be removed at a later stage
        m_out = mat[r, :].copy()

        # Insert the element that will be removed at the idx_in location
        m_out_r = m_out[r]

        for k in range(r, i - 1):
            m_out[k] = m_out[k + 1]
        m_out[idx_in - 1] = m_out_r

        # Shift blocks
        for k in range(r, i - 1):
            for j in range(mat.shape[1]):
                mat[k, j] = mat[k + 1, j]
        for j in range(r, i - 1):
            for k in range(mat.shape[0]):
                mat[k, j] = mat[k, j + 1]

        # Insert row/column
        mat[:, i - 1] = m_out
        mat[i - 1, :] = m_out

    return None


# %%

np.random.seed(123)
A = np.random.rand(7, 10)
A = A @ A.T
a = np.random.rand(7)
# for i in range(7, 0, -1):
#     A[:i, :] = i
#     A[:, :i] = i
i = 2

A_new = A.copy()
A_new[i, :] += a
A_new[:, i] += a

# print(np.linalg.inv(A_new))
# print(updateInverse(np.linalg.inv(A), a, i))

idx_in = 2
idx_out = 1

A0 = dropVariable(A, idx_in)
A0_inv = np.linalg.inv(A0)

A1 = dropVariable(A, idx_out)
A1_inv = np.linalg.inv(A1)


if idx_in > idx_out:
    # Store the row/column that will be removed at a later stage
    a_out = A0[idx_out, :].copy()

    # Insert the element that will be removed at the idx_in location
    a_idx_out = a_out[idx_out]

    for i in range(idx_out, idx_in - 1):
        a_out[i] = a_out[i + 1]
    a_out[idx_in - 1] = a_idx_out

    # Shift the center block one position towards the upper left part
    for i in range(idx_out, idx_in - 1):
        for j in range(idx_out, idx_in - 1):
            A0[i, j] = A0[i + 1, j + 1]

    # Shifting north
    for i in range(idx_out, idx_in - 1):
        # Shift the western block one position north
        for j in range(idx_out):
            A0[i, j] = A0[i + 1, j]
        # Shift the eastern block one position north
        for j in range(idx_in, A0.shape[1]):
            A0[i, j] = A0[i + 1, j]

    # Shifting west
    for j in range(idx_out, idx_in - 1):
        # Shift the northern block one position west
        for i in range(idx_out):
            A0[i, j] = A0[i, j + 1]
        # Shift the southern block one position west
        for i in range(idx_in, A0.shape[0]):
            A0[i, j] = A0[i, j + 1]

    # Move the row that will be removed later to the index where the row that
    # will be added will end up
    A0[:, idx_in - 1] = a_out
    A0[idx_in - 1, :] = a_out

    # Get the vector that updates A0 to A1
    update = A[idx_in, :] - A[idx_out, :]
    update[idx_in] = (A[idx_in, idx_in] - A[idx_out, idx_out]) / 2
    update = np.delete(update, idx_out)

    # Insert the row/column that is going to be removed at the index where the
    # row that will be added is going to be inserted
    removeAndInsertInplace(A0_inv, idx_out, idx_in)
    A0_inv = updateInverse(A0_inv, update, idx_in - 1)

print((A1_inv - A0_inv).round(5))
