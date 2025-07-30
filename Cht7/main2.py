import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchtext
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

# Test data
# Defining the Vocabulary:
word_to_ix = {"hello": 0, "world": 1}

embeds = nn.Embedding(2, 5)

# Printing Embedding Weights:
print('embeds.weight = ', embeds.weight)

lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
hello_embed = embeds(lookup_tensor)
print('hello_embed = ', hello_embed) # output "hello" weight.
