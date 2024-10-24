"""Proszę wybrać ulubiony język programowania, wygenerować
macierze losowe o wartościach z przedziału otwartego
(0.00000001, 1.0) i zaimplementować
1 Rekurencyjne odwracanie macierzy (10 punktów)
2 Rekurencyjna eliminacja Gaussa (10 punktów)
3 Rekurencyjna LU faktoryzacja (10 punktów)
4 Rekurencyjne liczenie wyznacznika (10 punktów)
Proszę zliczać liczbę operacji zmienno-przecinkowych
 (+-*/_liczb_) wykonywanych podczas mnożenia macierzy."""

import numpy as np
import sys
import os

# Dodaj ścieżkę katalogu wyżej (parent directory) do sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Teraz możesz zaimportować moduł z katalogu wyżej

from utility import (
    pad_matrix_even,
    unpad_matrix,
    get_random_matrix_pair_any_size,
    
)


def recursive_inverse(a):
    if np.size(a[0]) == 1:

        return a if a[0, 0] == 0 else np.array([[1 / a[0, 0]]])
    original_shape = a.shape
    a = pad_matrix_even(a)

    n = np.size(a[0])
    mid = n // 2

    a11 = a[:mid, :mid]
    a12 = a[:mid, mid:]
    a21 = a[mid:, :mid]
    a22 = a[mid:, mid:]

    a11inv = recursive_inverse(a11)
    s22 = a22 - np.linalg.multi_dot([a21, a11inv, a12])
    s22inv = recursive_inverse(s22)

    b11 = a11inv + np.linalg.multi_dot([a11inv, a12, s22inv, a21, a11inv])
    b12 = -np.linalg.multi_dot([a11inv, a12, s22inv])
    b21 = -np.linalg.multi_dot([s22inv, a21, a11inv])
    b22 = s22inv

    return unpad_matrix(
        np.vstack((np.hstack((b11, b12)), np.hstack((b21, b22)))), original_shape
    )


def recursive_Gauss():
    pass


def recursive_LU():
    pass


def recursive_determinant():
    pass


A, _ = get_random_matrix_pair_any_size(11)
print(A)
A_inv = recursive_inverse(A)
print(A_inv)
print(np.dot(A, A_inv))
