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
    pad_vector_even,
    matrice_vector_mult,
    get_random_vector_any_size,
    unpad_vector,
)
from Lab1.matrix_mult import strassen


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
    # s22 = a22 - np.linalg.multi_dot([a21, a11inv, a12])
    s22 = a22 - strassen(strassen(a21, a11inv), a12)
    s22inv = recursive_inverse(s22)

    # b11 = a11inv + np.linalg.multi_dot([a11inv, a12, s22inv, a21, a11inv])
    b11 = a11inv + strassen(
        strassen(strassen(strassen(a11inv, a12), s22inv), a21), a11inv
    )
    # b12 = -np.linalg.multi_dot([a11inv, a12, s22inv])
    b12 = -strassen(strassen(a11inv, a12), s22inv)
    # b21 = -np.linalg.multi_dot([s22inv, a21, a11inv])
    b21 = -strassen(strassen(s22inv, a21), a11inv)
    b22 = s22inv

    return unpad_matrix(
        np.vstack((np.hstack((b11, b12)), np.hstack((b21, b22)))), original_shape
    )


def recursive_LU(a):
    if np.size(a[0]) == 1:

        return np.array([[1]]), a
    original_shape = a.shape
    a = pad_matrix_even(a)

    n = np.size(a[0])
    mid = n // 2

    a11 = a[:mid, :mid]
    a12 = a[:mid, mid:]
    a21 = a[mid:, :mid]
    a22 = a[mid:, mid:]

    l11, u11 = recursive_LU(a11)
    u11inv = recursive_inverse(u11)
    l21 = np.dot(a21, u11inv)
    l11inv = recursive_inverse(l11)
    u12 = np.dot(l11inv, a12)
    # s = a22 - np.linalg.multi_dot([a21,u11inv,l11inv,a12])
    s = a22 - strassen(strassen(strassen(a21, u11inv), l11inv), a12)
    l22, u22 = recursive_LU(s)
    l = unpad_matrix(
        np.vstack((np.hstack((l11, np.zeros(l11.shape))), np.hstack((l21, l22)))),
        original_shape,
    )
    u = unpad_matrix(
        np.vstack((np.hstack((u11, u12)), np.hstack((np.zeros(l11.shape), u22)))),
        original_shape,
    )
    return l, u


def recursive_Gauss(a, b):

    original_shape_a = a.shape
    a = pad_matrix_even(a)
    n = np.size(a[0])
    mid = n // 2

    a11 = a[:mid, :mid]
    a12 = a[:mid, mid:]
    a21 = a[mid:, :mid]
    a22 = a[mid:, mid:]

    original_shape_b = b.shape
    print(b)

    b = pad_vector_even(b)

    b1 = b[:mid]
    b2 = b[mid:]

    l11, u11 = recursive_LU(a11)
    l11inv = recursive_inverse(l11)
    u11inv = recursive_inverse(u11)
    s = a22 - strassen(strassen(strassen(a21, u11inv), l11inv), a12)
    ls, us = recursive_LU(s)
    lsinv = recursive_inverse(ls)

    c11 = u11
    c12 = strassen(l11inv, a12)
    c21 = np.zeros(c12.shape)
    c22 = us

    lhs = unpad_matrix(
        np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22)))), original_shape_a
    )

    rhs1 = matrice_vector_mult(l11inv, b1)
    rhs2 = matrice_vector_mult(lsinv, b2) - matrice_vector_mult(
        strassen(strassen(strassen(lsinv, a21), u11inv), l11inv), b1
    )

    rhs = unpad_vector(np.hstack((rhs1, rhs2)), original_shape_b)

    return lhs, rhs


def recursive_determinant(a):
    l,u=recursive_LU(a)
    det=1
    for i in range(u.shape[0]):
        det*=u[i,i]
    return det


size = 11
A, _ = get_random_matrix_pair_any_size(size)

'''b = get_random_vector_any_size(size)

print(A, "\n", b, "\n")
# A_inv = recursive_inverse(A)
l, r = recursive_Gauss(A, b)
print(l, "\n", r, "\n")

print(np.linalg.solve(A,b),"\n",np.linalg.solve(l,r))

# print(np.dot(A, A_inv))'''

my_det=recursive_determinant(A)
np_det=np.linalg.det(A)
print(my_det, "\n", np_det, "\n")
