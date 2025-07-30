import torch
x = torch.randn((1,4),dtype=torch.float32,requires_grad=True)
y = x ** 2
z = y * 4
output = z.mean()
print(x)
print(y)
print(z)
print(output)
output.backward()

from torchviz import make_dot

dot = make_dot(output)
dot.view()  # Displays the calculated graph in the default graphical viewer
