import torch
torch.manual_seed(16)

loss = torch.nn.MSELoss()  # Generate MSE loss object
# 3 x 5 matrix
inp = torch.randn(3, 5, requires_grad=True)
print(inp)
target = torch.randn(3, 5) # target values
print(target)
output = loss(inp, target) # Evaluate the mean square error between input and target
print(output)
