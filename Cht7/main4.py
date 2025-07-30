import torch
import torch.nn as nn
rnn = nn.RNN(10, 20)
# RNN processing
# Test data
input = torch.randn(5, 10) # 5 列 10 行 input (10 維的資料我有 5 筆)
output, hn = rnn(input) # 丟進去 rnn model.
print(output.shape, hn.shape)
print(output)
print('-'*100)
print(hn)
