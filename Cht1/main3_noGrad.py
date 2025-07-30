import torch
x = torch.tensor(4.0, requires_grad=True)
y = x+1 # y = 5
z = 2*x # z = 8
# 注意: 只針對 y 做 no grad
with torch.no_grad():
    y = y*y  # y = 25 (沒了)
u = y+z # u = 33
u.backward()       # Back propogation
print(u)
print(x.grad)      # the gradient of x
