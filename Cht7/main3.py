import torch
import torch.nn as nn
vocab_size = 1027
hidden_size = 256 # hidden layer size is 256
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=hidden_size)

from torchinfo import summary
print(summary(rnn_layer))
print()
