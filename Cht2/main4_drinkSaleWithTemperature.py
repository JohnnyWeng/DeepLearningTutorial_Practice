import numpy as np
import torch

torch.manual_seed(16)

def train(X, y, epochs=100, lr=0.0001353):
    loss_list, w_list, b_list = [], [], []
    w = torch.randn(1, requires_grad=True, dtype=torch.float)
    b = torch.randn(1, requires_grad=True, dtype=torch.float)

    for epoch in range(epochs):
        y_pred = w * X + b
        MSE = torch.square(y - y_pred).mean()
        MSE.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
    if (epoch + 1) % 1000 == 0:
        w_list.append(w.item())  # w.item()：To constant
        b_list.append(b.item())
        loss_list.append(MSE.item())
    w.grad.zero_()
    b.grad.zero_()
    return w_list, b_list, loss_list
temperatures = np.array([29, 28, 34, 31, 25, 29, 32, 31,
                         24, 33, 25, 31, 26, 30])
drink_sales = np.array([7.7, 6.2, 9.3, 8.4, 5.9, 6.4, 8.0, 7.5,
                        5.8, 9.1, 5.1, 7.3, 6.5, 8.4])

# training
w_list, b_list, loss_list = train(torch.tensor(temperatures), torch.tensor(drink_sales), epochs=100000)

print(f'w={w_list[-1]}, b={b_list[-1]}')
print('when temperature=26, the predicted sale is ', 26*w_list[-1] + b_list[-1])
print('when temperature=30, the predicted sale is ', 30*w_list[-1] + b_list[-1])

import matplotlib.pyplot as plt
plt.scatter(temperatures, drink_sales, label='data')
plt.plot(temperatures, w_list[-1] * temperatures + b_list[-1], 'r-', label='predicted')
plt.legend()
plt.show()
