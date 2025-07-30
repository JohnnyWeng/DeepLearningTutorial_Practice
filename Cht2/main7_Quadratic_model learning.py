import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
torch.manual_seed(16) #for fixed model

# Generate Quadratic Data
def generate_quadratic_data(num_samples):
    x = torch.linspace(-5, 5, num_samples) #  creates x values evenly spaced between -5 and 5.
    # y = 2x^2 + 3x + 1 + noise
    y = 2 * x**2 + 3 * x + 1 + torch.randn(num_samples) * 2
    return x, y

# 31 data to train.
model = torch.nn.Sequential(
    # input is 1, 1~10
    #  The 1st layer has 10 neurons
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    # output is also the 1
    torch.nn.Linear(10, 1)
)

# Another way to define the mdoel: Define a simple neural network model
# The above one is the same as the below one.
class QuadraticModel(nn.Module):
    # definition
    def __init__(self):
        super(QuadraticModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input layer to hidden layer
        self.fc2 = nn.Linear(10, 1)  # Hide layer to output layer

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

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_samples = 100
x, y = generate_quadratic_data(num_samples)
print('x = ', x)
print('y = ', y)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(x.unsqueeze(1)) #Increase dimensions
    print('x = ', x)
    print('x.unsqueeze(1) = ', x.unsqueeze(1))
    print(' outputs = ', outputs)
    loss = criterion(outputs, y.unsqueeze(1)) # Increase dimensions
    # or loss = criterion(outputs.squeeze(), y) # 壓縮: reduce 1 dimension.
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # update weight of the model.

    if (epoch + 1) % 100 == 0:
        print('model.params = ', model.parameters())
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the fitted curve
model.eval()
with torch.no_grad():
    x_test = torch.linspace(-5, 5, 100)
    y_pred = model(x_test.unsqueeze(1))
    print('y_pred = ', y_pred)

plt.figure(figsize=(8, 6))
plt.scatter(x.numpy(), y.numpy(), label='Real data') # blue points
print('x_test.numpy() = ', x_test.numpy())
print('y_pred.numpy() = ', y_pred.numpy())
plt.plot(x_test.numpy(), y_pred.numpy(), 'r-', label='Fit curve') # red line: predict
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
