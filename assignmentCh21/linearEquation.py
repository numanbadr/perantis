import numpy as np

A = np.array([[2, 1, 1], [1, 1, 0], [0, 1, -3]])

b = np.array([2, 2, 1])

ans = np.linalg.solve(A, b)

print(f'x = {ans[0]}\ny = {ans[1]}\nz = {ans[2]}')
