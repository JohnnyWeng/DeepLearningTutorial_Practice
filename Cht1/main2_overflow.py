import numpy as np
import matplotlib.pyplot as plt

def func(x): return 2*x**4-3*x**2+2*x

def dfunc(x): return 8*x**3-6*x+2

# Gradient Descent
def GD(x_start, df, epochs, lr):
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x

    for i in range(epochs):
        print(x)
        dx = df(x)
        x -= dx * lr
        xs[i+1] = x
    return xs
# (Hyperparameters)
x_start = 5     # initial weight
epochs = 15
lr = 0.3        # learning rate too big
# lr = 0.01
w = GD(x_start, dfunc, epochs, lr=lr)
print(w)
