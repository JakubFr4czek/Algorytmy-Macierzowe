from utility import pad_matrix_even, get_random_matrix_pair_any_size, unpad_matrix
from matrix_mult import binet
import numpy as np


A, B = get_random_matrix_pair_any_size(5)



C1 = binet(A, B)

print(C1)


C2 = A@B
print(C2)
