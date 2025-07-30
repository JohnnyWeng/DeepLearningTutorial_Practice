import torch

m = torch.nn.ReLU()
input = torch.tensor([5, 2, 0, -10])
output = m(input)
print('ReLU= ', output)
m = torch.nn.Softmax(dim=1)
input = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=float)
output = m(input)
print('ReLU = ',output)

import torch
m = torch.nn.Sigmoid()
input = torch.tensor([5, 2, 0, -10, -100], dtype=float)
output = m(input)
print('Sigmoid= ', output)

# dim=1 (The data for each column is converted into a probability distribution)
m = torch.nn.Softmax(dim=1)
input = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=float)
output = m(input)
print('Softmax = ',output)

# Tanh
m = torch.nn.Tanh() # Default dim = 0
input = torch.tensor([5, 2, 0, -10, -100], dtype=float)
output = m(input)
output = m(input)
print('Tanh = ',output)
