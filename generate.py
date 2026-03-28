import numpy as np
import sys
import os


def save_matrix(n: int, filename: str, matrix: np.ndarray):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(' '.join(f"{val:.6f}" for val in row) + "\n")
    print(f"File {filename} was created")


if __name__ == "__main__":
    SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print(f"Generating {SIZE} x {SIZE} matrices...")
    matrix_a = np.random.uniform(1, 10, (SIZE, SIZE))
    matrix_b = np.random.uniform(1, 10, (SIZE, SIZE))
    save_matrix(SIZE, "data/matrix_a.txt", matrix_a)
    save_matrix(SIZE, "data/matrix_b.txt", matrix_b)

    print("Done!")