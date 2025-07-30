import torch
import torch.nn as nn

# Define a 2x2 input feature map (tensor)
input_tensor = torch.tensor([[1, 2],
                             [3, 4]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
# Shape after unsqueeze: (1, 1, 2, 2) to fit PyTorch's expected (batch_size, channels, height, width)

deconv = nn.ConvTranspose2d(
    in_channels=1,      # Number of input channels
    out_channels=1,     # Number of output channels
    kernel_size=2,      # Size of the convolutional kernel: 2x2
    stride=2,           # Stride value to expand the output size
    bias=False          # Disable bias term, Do not calculate the bias
)

# Manually set the weight of the transposed convolution kernel to a simple pattern
with torch.no_grad():
    deconv.weight = nn.Parameter(torch.tensor([[[[1, 0],
                                                 [0, 1]]]], dtype=torch.float32))

output_tensor = deconv(input_tensor)

# Print input and output details
print("Original input feature map:")
print(input_tensor[0][0])

print("Kernel:")
print(deconv.weight[0][0])

print("\nOutput feature map after transposed convolution:")
print(output_tensor[0][0])
