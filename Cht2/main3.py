import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
torch.manual_seed(16) #for fixed random data
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Generate random data for a quadratic function
def generate_quadratic_data(num_samples):
    x = torch.linspace(-5, 5, num_samples)
    y = 2 * x**2 + 3 * x + 1 + torch.randn(num_samples) * 2
    return x, y

# Define a quadratic function model
def quadratic_model(x, params):
    a, b, c = params
    return a * x**2 + b * x + c

# Define the loss function (mean squared error)
def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred)**2)


def gradient_descent(x, y, learning_rate, num_epochs):
    # Initialize model parameters randomly
    params = torch.randn(3, requires_grad=True)

    for epoch in range(num_epochs):
        # Calculate the model predictions
        y_pred = quadratic_model(x, params)

        # Calculate the loss function
        loss = mean_squared_error(y, y_pred)
        # Calculate gradients and update parameters
        loss.backward()
        params.data -= learning_rate * params.grad.data
        # Clear the gradient
        params.grad.zero_()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return params
