import torch
from torchviz import make_dot

# Create a simple model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(2, 2) # input size 2 and output size 2
        self.fc2 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create input data
input_data = torch.randn(1, 2) #1 sample with 2 features

# Create a model instance
model = SimpleModel()

# Forward propagation
output = model(input_data)

# Use make_dot functions to visualize the computational graph
dot = make_dot(output, params=dict(model.named_parameters()))
dot.view()  # Displays the calculated graph in the default graphical viewer
