import numpy as np
from scipy.signal import convolve2d

A = np.array([[1, 2, 3, 2, 1],
              [0, 5, 1, 1, 6],
              [4, 0, 2, 2, 2],
              [1, 2, 1, 1, 3],
              [1, 1, 5, 2, 2]], dtype=int)

V = np.array([[1, 3, 1], [0, 3, 0], [1, 0, 2]], dtype=int)

# You can direclty use padding on the convolve2d function.
print(convolve2d(A, V, mode='valid')) # become smaller
print(convolve2d(A, V, mode='same')) # retain the same size.
print(convolve2d(A, V, mode='full')) # complement 2 layers of 0.
