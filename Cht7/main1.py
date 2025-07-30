import torch
# Defining Word Index Tensors:
word1 = torch.LongTensor([0, 1, 100])
word2 = torch.LongTensor([2, 99, 3])

embedding = torch.nn.Embedding(101, 5)

print('embedding.weight = ', embedding.weight)

# Retrieving Embeddings for word1 and word2:
print('word1:')  # 0~100 向量組合
print(embedding(word1))
print('word2:')
print(embedding(word2))
