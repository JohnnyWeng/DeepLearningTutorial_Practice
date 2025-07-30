import torch
def dfunc(x):
    x = torch.tensor(float(x), requires_grad=True)
    y = x ** 0.5
    y.backward() # 微分 y
    # x dx/dy
    return x.grad
print(dfunc(100))
