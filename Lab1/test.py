from utility import pad_matrix_even, pad_matrix_to_2nd_power, get_random_matrix_pair_any_size, unpad_matrix
from matrix_mult import strassen2, strassen
import numpy as np

A, B = get_random_matrix_pair_any_size(46)

C1 = strassen2(A, B)

original_shape = A.shape

A = pad_matrix_to_2nd_power(A)
B = pad_matrix_to_2nd_power(B)

C2 = strassen(A, B)

C2 = unpad_matrix(C2, original_shape)

print(np.allclose(C1, C2, atol=1e-5))