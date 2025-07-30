import torch
def dfunc(x):
    x = torch.tensor(float(x), requires_grad=True)
    y = x ** 2 + 1
    z = y ** 5
    z.backward() # dz/dx
    return x.grad
print(dfunc(1)) # (dz/dx)|x=1
