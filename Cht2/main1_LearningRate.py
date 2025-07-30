import numpy as np
import torch
torch.manual_seed(16)
def train(X, y, epochs=100, lr=0.00000001):
    loss_list, w_list, b_list = [], [], []
    w = torch.randn(1, requires_grad=True, dtype=torch.float)
    b = torch.randn(1, requires_grad=True, dtype=torch.float)
    for epoch in range(epochs):
        y_pred = w * X + b
        # loss funciton: MSE
        MSE = torch.square(y - y_pred).mean()
        MSE.backward() # 微分 MSE.
        with torch.no_grad(): # do not calculate the gradient
            w -= lr * w.grad
            b -= lr * b.grad
            if epoch == 1: # check the w and b
                print('w, b =', w, b)
        # Store the value for every 1000 times
        if (epoch + 1) % 1000 == 0 or epochs < 1000:
            w_list.append(w.item())  # w.item()：To constant
            b_list.append(b.item())
            loss_list.append(MSE.item())
        # Clear out the gradient so that they don't accumulate.
        w.grad.zero_()
        b.grad.zero_()

    return w_list, b_list, loss_list

n = 100
# 0~50 之間,平均產生 100 points
X = np.linspace(0, 50, n)
y = np.linspace(0, 50, n)
np.random.seed(1)
# Add noise, 稍微讓線有變化
# 上下隨機增加 1
X += np.random.uniform(-1, 1, n) # Training data
y += np.random.uniform(-1, 1, n) # validating data

w_list, b_list, loss_list = train(torch.tensor(X), torch.tensor(y), epochs=100000)

print(f'w={w_list[-1]}, b={b_list[-1]}')

import matplotlib.pyplot as plt
plt.scatter(X, y, label='data')
plt.plot(X, w_list[-1] * X + b_list[-1], 'r-', label='predicted')
plt.legend()
print(loss_list)
# draw the line
plt.plot(loss_list)
plt.show()
