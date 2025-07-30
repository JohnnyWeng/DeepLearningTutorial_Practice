import numpy as np
import pandas as pd
from sklearn import datasets
import torch

dataset = datasets.load_iris() # from sklearn
# organize the dataset
df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
print('df.head()= \n', df.head())

from sklearn.model_selection import train_test_split
# test_size=0.2. training dataset:validation dataset (8:2). training dataset: 20%
X_train, X_test, y_train, y_test = train_test_split(df.values,
                                        dataset.target, test_size=0.2)

y_train_encoding = pd.get_dummies(y_train)
y_test_encoding = pd.get_dummies(y_test)

print('y_train.shape = ', y_train.shape)
print('y_train_encoding.shape = ', y_train_encoding.shape) # 3
# Convert to tensor
X_train = torch.FloatTensor(X_train)
y_train_encoding = torch.FloatTensor(y_train_encoding.values)
X_test = torch.FloatTensor(X_test)
y_test_encoding = torch.FloatTensor(y_test_encoding.values)
print('X_train.shape, y_train_encoding.shape = ', X_train.shape, y_train_encoding.shape)

import pandas as pd
# Build a DataFrame
data = { 'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red'] }
# 1d
df = pd.DataFrame(data) # conver the array into structured, table-like format
print('df = ', df)
dummies = pd.get_dummies(df['Color'])
print('dummies = \n', dummies)

model = torch.nn.Sequential(
    torch.nn.Linear(4, 3),
    torch.nn.Softmax(dim=1)
)

# This will compute the sum of all the squared errors. Instead of averaging
loss_function = torch.nn.MSELoss(reduction='sum')
# We use Adam with learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs=10000
accuracy = []
losses = []
for i in range(epochs):
    y_pred = model(X_train)
    loss = loss_function(y_pred, y_train_encoding)
    # argmax: Since y_pred all about possibilities(90%, 50%...), therefore we want to know the highest one so that we know it is 0,1,2 of the type of the iris flower.
    accuracy.append((np.argmax(y_pred.detach().numpy(), axis=1) == y_train)
                    .sum()/y_train.shape[0]*100)
    losses.append(loss.item())
    # The below 3 lines we always add.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print out every 100 times
    if i%100 == 0:
        print('loss = ', loss.item())

    import torch

    tensor = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]])

    max_values, indices = torch.max(tensor, dim=1)

    print("Max: ", max_values)
    print("Index of max: ", indices)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
# 1 列兩行的第一個
plt.subplot(1, 2, 1)
plt.title('loss:', fontsize=20)
plt.plot(range(0, epochs), losses)
# 1 列兩行的第二個
plt.subplot(1, 2, 2)
plt.title('accuracy:', fontsize=20)
plt.plot(range(0, epochs), accuracy)
plt.ylim(0, 100)
plt.show()

'''
Predict test is also the tensor.
tensor([[4.0500e-03, 9.9595e-01, 2.6683e-10],
        [9.9986e-01, 1.4242e-04, 0.0000e+00],
        [3.8670e-18, 4.5751e-12, 1.0000e+00],                  .
'''
predict_test = model(X_test)
_, y_pred = torch.max(predict_test, 1) # 最大值

# We've split the data before.
print('y_test.shape[0] = ',  y_test.shape[0]) # 30 data
print(f'test data accuracy: {((y_pred.numpy() == y_test).sum() / y_test.shape[0]):.2f}') # 0.93
