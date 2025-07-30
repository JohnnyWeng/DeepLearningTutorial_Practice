import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
torch.manual_seed(16) #for fixed model
def generate_quadratic_data(num_samples):
    x = torch.linspace(-5, 5, num_samples)
    y = x ** 3 - 5 * x ** 2 + x - 10 + torch.randn(num_samples) * 2  # Add noise
    return x, y

class QuadraticModel(nn.Module):
    # definition
    def __init__(self):
        super(QuadraticModel, self).__init__()
        # 1: input of the neurons
        # 50: output of the neurons
        self.fc1 = nn.Linear(1, 50)  # Input layer to hidden layer
        self.fc2 = nn.Linear(50, 1)  # Hide layer to output layer

    def forward(self, x):
        # relu is not a layer. It is for the calculation
        # The below code is combining two lines of code: relu + fc
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Create a model
model = QuadraticModel()

# Create an optimizer
criterion = nn.MSELoss()  # Mean squared error loss function

optimizer = torch.optim.Adam(model.parameters(), lr=0.075)

num_samples = 100
x, y = generate_quadratic_data(num_samples)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward propagation
    outputs = model(x.unsqueeze(1)) #Increase dimensions
    print(' outputs = ', outputs)
    loss = criterion(outputs, y.unsqueeze(1)) #Increase dimensions
    # or loss = criterion(outputs.squeeze(), y) # 壓縮: reduce 1 dimension.
    # Backpropagation and optimization
    optimizer.zero_grad()
    print('optimizer = ', optimizer)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the fitted curve
model.eval()
with torch.no_grad():
    x_test = torch.linspace(-5, 5, 100)
    y_pred = model(x_test.unsqueeze(1))


plt.figure(figsize=(8, 6))
plt.scatter(x.numpy(), y.numpy(), label='Real data')
plt.plot(x_test.numpy(), y_pred.numpy(), 'r-', label='Fit curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
