import numpy as np
import matplotlib.pyplot as plt

def func(x): return x ** 2

def dfunc(x): return 2 * x

x_start = 5     # Initial weight
epochs = 15
lr = 0.3        # learning rate

# Gradient Descent
def GD(x_start, df, epochs, lr):
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x # save the initial weight
    # 繞 15 次: Calculate the weight
    for i in range(epochs):
        dx = df(x) # Our dfunc Model
        # Update x_new = x — learning_rate * gradient
        x += - dx * lr
        xs[i+1] = x # Save all the weight.
    return xs

# Gradient Decendent + weight
w = GD(x_start, dfunc, epochs, lr=lr)
print ('weight = ', np.around(w, 2))

t = np.arange(-6.0, 6.0, 0.01)
# Blue line: Plot the Function -> y=x ^ 2:
plt.plot(t, func(t), c='b')

plt.plot(w, func(w), c='r', marker ='o', markersize=5)

plt.xlabel('X', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.show()
