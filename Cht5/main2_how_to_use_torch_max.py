import os

import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "True "
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor = torch.tensor([[1, 3, 2], [5, 2, 8]])
max_values, indices = torch.max(tensor, dim=1)
print(max_values)  # Output: tensor([3, 8]) -> find each max value in each of the array.
print(indices)     # Output: tensor([1, 2]) -> find the index of those 2 values.
