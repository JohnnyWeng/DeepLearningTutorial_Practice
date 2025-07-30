import torch
torch.manual_seed(1)
x = torch.ones(5)  # input tensor
# y 是 3 個點的向量
y = torch.zeros(3)  # expected output, Ground truth
# 產生亂數: 大部分 -1 ~1 之間, 可微分的
w = torch.randn(5, 3, requires_grad=True)
#  bias
b = torch.randn(3, requires_grad=True)
# multiply x and w -> x * w (兩個向量做相乘)
t = torch.matmul(x, w) # Finally (3, 1) squeezes to (3, )
z = t+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print('x = ', x) # 1d array
print('y = ', y) # blank array with 0 zero
print('w = ',w) # 2D, 5 rows, 3 column array.
print('b = ',b)
print('t = ',t)
print('z = ',z)
print('loss = ',loss)

