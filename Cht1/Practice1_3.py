import torch
from torchviz import make_dot
import numpy as np

x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)

# Define the operations
y = x ** 4             # y = x^4
z = 2 * y              # z = 2 * y
u = z.mean()           # u = mean(z)

# Print the values to verify
print(f"x: {x}")
print(f"y (x^4): {y}")
print(f"z (2*y): {z}")
print(f"u (mean(z)): {u}")

# Backpropagate to compute gradients
u.backward()

# Visualize the computational graph
dot = make_dot(u, params={"x": x})
dot.view()  # Displays the graph in the default graphical viewer
