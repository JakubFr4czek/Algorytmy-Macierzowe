import numpy as np
import math

lower_bound = 0.00000001
upper_bound = 1.0
epsilon = 1e-10


def get_random_matrices_power_of_2(n=1):

    # creating different size matrices, two of each with values in the open range (0.00000001, 1.0)
    matrices = []
    for i in range(n):
        matrices.append(
            [
                np.random.uniform(
                    lower_bound + epsilon, upper_bound, (2 ** (i + 1), 2 ** (i + 1))
                ),
                np.random.uniform(
                    lower_bound + epsilon, upper_bound, (2 ** (i + 1), 2 ** (i + 1))
                ),
            ]
        )

    return matrices


def get_random_matrices_any_size(n=1):

    # creating different size matrices, two of each with values in the open range (0.00000001, 1.0)

    matrices = []
    for i in range(n):
        matrices.append(
            [
                np.random.uniform(lower_bound + epsilon, upper_bound, (i + 1, i + 1)),
                np.random.uniform(lower_bound + epsilon, upper_bound, (i + 1, i + 1)),
            ]
        )

    return matrices


def get_random_matrix_pair_any_size(n=1):
    return np.random.uniform(
        lower_bound + epsilon, upper_bound, (n, n)
    ), np.random.uniform(lower_bound + epsilon, upper_bound, (n, n))

def get_random_vector_any_size(n):
    return np.random.uniform(lower_bound + epsilon, upper_bound,n)



def pad_matrix_even(m):
    if m.shape[0] % 2 != 0:
        m = np.vstack((m, np.zeros(shape=(1, m.shape[1]))))

    if m.shape[1] % 2 != 0:
        m = np.hstack((m, np.zeros(shape=(m.shape[0], 1))))

    return m

def matrice_vector_mult(m,v):
    res = np.zeros(m.shape[0])
    for i in range(m.shape[0]):
        for j in range(v.shape[0]):
            res[i]+=m[i,j]*v[j]
    return res

def pad_vector_even(v):

    if v.shape[0] % 2 != 0:
        v = np.hstack((v, np.zeros(1)))

    return v


def pad_matrix_to_nxn_shape(m, n):

    rows, cols = m.shape

    padded_matrix = np.zeros((n, n))

    padded_matrix[:rows, :cols] = m

    return padded_matrix


def unpad_matrix(m, prev_shape):
    m = m[: prev_shape[0], : prev_shape[1]]
    return m

def unpad_vector(v,prev_shape):
    v=v[:prev_shape[0]]
    return v

def get_AI_matrices():

    # matrices for AI algorithm (4x5)x(5x5)
    m45 = np.random.uniform(lower_bound + epsilon, upper_bound, (4, 5))
    m55 = np.random.uniform(lower_bound + epsilon, upper_bound, (5, 5))

    return m45, m55
