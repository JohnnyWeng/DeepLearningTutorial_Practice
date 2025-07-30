import torch
from torch import nn
torch.manual_seed(16) #for fixed model

# Build a 1D tensor
x = torch.tensor([1, 2, 3, 4])
print("Original tensor:", x) # x is a just tensor if you print it out.
print("Original shape:", x.shape) # torch.Size([4])
print("Third value:", x[1])
print("Forth value:", x[3])

# Use unsqueeze to "add" a new dimension "in dimension 0"
x_unsqueezed = x.unsqueeze(0) #1d -> 2d

print("Tensor after unsqueeze at dim 0:", x_unsqueezed)
print("Shape after unsqueeze at dim 0:", x_unsqueezed.shape) # torch.Size([1, 4]) -> 1d with 4 values in each dimentaions

# Use unsqueeze to add a new dimension "in dimension 1" -> add the dimension in the deepest matrix
x_unsqueezed_1 = x.unsqueeze(1)

print("Tensor after unsqueeze at dim 1:", x_unsqueezed_1)
print("Shape after unsqueeze at dim 1:", x_unsqueezed_1.shape)

#Define a simple neural network model
class QuadraticModel(nn.Module):
    def __init__(self):
        super(QuadraticModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input layer to hidden layer
        # 10 (weights) + 1 (bias) = 11 parameters
        self.fc2 = nn.Linear(10, 1)  # Hide layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Create a model
model = QuadraticModel()

from torchinfo import summary
print(summary(model))
