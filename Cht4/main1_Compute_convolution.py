import numpy as np

# 1x1
source_map = np.array(list('1110001110001110011001100')).astype(int)
print('source_map = ', source_map)
# 1d resshape to 5x5 array.
source_map = source_map.reshape(5,5)
print('source：')
# You can see it becomes 5x5 instead of 1x1.
print(source_map)
print('-------------------------------------------------------------------')

filter1 = np.array(list('101010101')).astype(int).reshape(3,3)
print('filter:\n', filter1)

'''
Compute convolution: windows sliding.

source：
[[1 1 1 0 0]
 [0 1 1 1 0]
 [0 0 1 1 1]
 [0 0 1 1 0]
 [0 1 1 0 0]]

filter:
 [[1 0 1]
 [0 1 0]
 [1 0 1]]
 
'''
width = height = source_map.shape[0] - filter1.shape[0] + 1
result = np.zeros((width, height))

for i in range(width):
    for j in range(height):
        # 內積相乘, 非矩陣相乘。兩兩相乘, 再相加
        value1 =source_map[i:i+3, j:j+3] * filter1
        result[i, j] = np.sum(value1)
print('compute convolution:')
print(result)

from scipy.signal import convolve2d
print("Using scipy:\n", convolve2d(source_map, filter1, mode='valid'))
