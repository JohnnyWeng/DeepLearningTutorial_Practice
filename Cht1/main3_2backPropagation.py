import torch

x = torch.tensor(2.0, requires_grad=True)
y = 2*x+1 # y = 5
z = x*y # z = 2x5 = 10
with torch.no_grad():
    z = y*z # z = 5x10 = 50
u = y*z # u = 5x50 = 250
u.backward()       # Back propogation
print(u)
print(x.grad)      # the gradient of x
