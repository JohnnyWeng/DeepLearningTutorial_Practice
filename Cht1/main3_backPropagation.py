import torch
x = torch.tensor(4.0, requires_grad=True)
y = x+1 # y = 5
z = 2*x # z = 8
with torch.no_grad():
    y = y*y # y = 25
y.backward()       # Back propogation
print(y)
print(x.grad)      # the gradient of x
