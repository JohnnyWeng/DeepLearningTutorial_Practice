from torch import nn

# Sequential model: sequential executing for multiple layers.
model = nn.Sequential(
          # input 256, output:20
          nn.Linear(256,20), # ppt: (256 + 1) x 20
          nn.ReLU(),
          nn.Linear(20,64), # ppt: (20+1)x64
          nn.ReLU(),
          nn.Softmax(dim=1),
        )

from torchinfo import summary
print(summary(model, (1, 256)))
print(summary(model, (76, 256)))
