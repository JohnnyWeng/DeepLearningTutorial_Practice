import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


PATH_DATASETS = ""
BATCH_SIZE = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# download MNIST training data at the same directory with current program file
train_ds = MNIST(PATH_DATASETS, train=True, download=True,
                 transform=transforms.ToTensor())

# download MNIST test data
test_ds = MNIST(PATH_DATASETS, train=False, download=True,
                 transform=transforms.ToTensor())

# the shape of training data and test data
print(train_ds.data.shape, test_ds.data.shape)
print(train_ds.targets.shape, test_ds.targets.shape)
# :10] -> 0~9
print(train_ds.targets[:10]) # show the first 10 data. tensor([5, 0, 4, 1, 9, 2, 1, 3, 1, 4])

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import matplotlib.pyplot as plt
X = train_ds.data[0] # without normalization
print('x = ', X)
# 60000 training data that are 28x28
# convert to 28x28 in gray scale.
plt.imshow(X.reshape(28,28), cmap='gray')
plt.show()

# Create model
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    # output: 256 neurons
    torch.nn.Linear(28 * 28, 256),
    # 0~9 numbers.
    # connect to the previous output: 256 neuron.
    # output: 0~9, 10 的數字。
    torch.nn.Linear(256, 10),
# You can directly to insert into device here. You don't need to do it again.
).to(device)

epochs = 6
lr=0.1
# Create DataLoader
train_loader = DataLoader(train_ds, batch_size=600)
# Set optimizer
optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
model.train() # set the model to training mode
loss_list = []
for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        # 微分
        loss.backward()
        # update the attributes
        optimizer.step()
        if batch_idx % 10 == 0:
            loss_list.append(loss.item())
            batch = batch_idx * len(data)
            data_count = len(train_loader.dataset)
            percentage = (100. * batch_idx / len(train_loader))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)' +
                  f'  Loss: {loss.item():.6f}')

import matplotlib.pyplot as plt
print('loss_list = ', loss_list)
plt.plot(loss_list, 'r')
plt.show()
# shuffle -> 資料要不要洗牌, 打亂
test_loader = DataLoader(test_ds, shuffle=False, batch_size=test_ds.targets.shape[0])

model.eval()
test_loss = 0
correct = 0
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    output = model(data)

    # sum up batch loss
    test_loss += criterion(output, target).item()
    # output: The output of the prediction. argmax: get the highest prediction
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
# normalize
test_loss /= len(test_loader.dataset)
batch = batch_idx * len(data)
data_count = len(test_loader.dataset)
percentage = 100. * correct / data_count
# actual    :  [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
# prediction:  [7 2 1 0 4 1 4 9 6 9 0 6 9 0 1 5 9 7 3 4]
# 5 -> 6 (wrong)
print(f'Average_loss: {test_loss:.4f}, correct_rate: {correct}/{data_count}' +
      f' ({percentage:.0f}%)\n') # 91% -> Every 10 images will have a wrong one.

predictions = []
with torch.no_grad():
    for i in range(20):
        # data: pixel value of the image, target: label (the digit shown in the image)
        data, target = test_ds[i][0], test_ds[i][1]
        if i ==1: # See what are those params are.
            print('data = ', data)
            print('target = ', target) #2
        data = data.reshape(1, *data.shape).to(device) # *:Unpack the list into independent parameters
        output = torch.argmax(model(data), axis=-1)
        if i == 1:
            print('output = ', output) # The output represents the predicted label (a digit from 0 to 9) for a given image in the test dataset. output = tensor([2], device='cuda:0') means the model predicted the digit 2 for the image.
        predictions.append(str(output.item()))

print('actual    :', test_ds.targets[0:20].numpy())
print('prediction: ', ' '.join(predictions[0:20]))

import numpy as np

i=8
data = test_ds[i][0]
data = data.reshape(1, *data.shape).to(device) # *:Unpack the list into independent parameters

#print(data.shape)
predictions = torch.softmax(model(data), dim=1)
print(f'0~9 predict rate: {np.around(predictions.cpu().detach().numpy(), 2)}')
print(f'0~9 predict rate: {np.argmax(predictions.cpu().detach().numpy(), axis=-1)}') # This prints the predicted digit by finding the index of the highest probability

torch.save(model, 'model.pt')
# load model
model = torch.load('model.pt')
# save weights
torch.save(model.state_dict(), 'model.pth')
# model.state_dict(), using dictionary to represent parameters of model
# load weights
model.load_state_dict(torch.load('model.pth'))
# show state_dict of each dimension
# You can print in this way ---- for a code block in the console.
print("----------------------------------")
print("The state_dict of each layer:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
print("----------------------------------")
'''
Hand-written Digits recognition (8)
'''
from skimage import io
from skimage.transform import resize
import numpy as np
print("[Start] Image recognition")
for i in range(10):
    uploaded_file = f'./myDigits/{i}.png'
    image1 = io.imread(uploaded_file, as_gray=True)
    # resize to (28, 28) -> the input of model is 28 x 28, so you need to modify .
    image_resized = resize(image1, (28, 28), anti_aliasing=True)
    X1 = image_resized.reshape(1,28, 28)

    # reverse to color (255->0, 0->255)
    X1 = torch.FloatTensor(1.0-X1).to(device)

    # predict
    predictions = torch.softmax(model(X1), dim=1)
    print(f'actual/prediction: {i}{np.argmax(predictions.detach().cpu().numpy())}')

print('model = ', model)
# Show summary of model
for name, module in model.named_children():
    print(f'{name}: {module}')

from torchinfo import summary
print(summary(model, (60000, 28, 28))) # input dimension
