import numpy as np
import sys
import os


def checkMultiply(matrix_a: str, matrix_b: str, matrix_c: str):
    A = np.loadtxt(matrix_a, skiprows=1)
    B = np.loadtxt(matrix_b, skiprows=1)
    C = np.loadtxt(matrix_c, skiprows=1)

    true_C = A@B
    if np.allclose(true_C, C):
        print("Verification completed. The result is correct!")
    else:
        print("Verification completed. The result IS NOT CORRECT")


if __name__ == "__main__":
    checkMultiply(sys.argv[1], sys.argv[2], sys.argv[3])