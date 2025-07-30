import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
torch.manual_seed(16) #for fixed random data
a = torch.tensor([5.0], requires_grad=True)
b = a * 2
c = a * 2
print(a)
print(b)
print(c)

b.add_(2)
c.add_(2)
print(b)
print(c)

with torch.no_grad():
    b.mul_(2) # without tracking
c.mul_(2)
print('Gradient b = ',b)
print('Gradient c = ',c)

b.backward() # b = 2a + 2
print(a.grad) # b’(a)
a.grad.zero_()
c.backward() # c = 4a + 4
print(a.grad) # c’(a)
