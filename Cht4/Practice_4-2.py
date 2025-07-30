import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
torch.manual_seed(16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transform: normalize the data
transform = transforms.Compose(
    [transforms.ToTensor(),
     # Read image ranged from [0, 1] and transform to [-1, 1], x-mean/std
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     # ImageNet -> for imagenet, the result will be better.
     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

batch_size = 1000

train_ds = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_ds = torchvision.datasets.CIFAR10(root= './CIFAR10', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
print(train_ds.data.shape, test_ds.data.shape)

# 10 classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
import matplotlib.pyplot as plt
import numpy as np

fruit = ["apple","banana","lemon"]
for i in fruit:
    print(i)
fr = iter(fruit)

for i in range(3):
    print(next(fr)) # print the fruit
print("-"*10) # print ---------------
for i in fr:
    print(i)


# Image display function
def imshow(img):
    img = img * 0.5 + 0.5  # Restore the image
    npimg = img.numpy()
    # The color is shifted to the last dimension
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # (a, b, c) 🡪 (b, c, a)
    plt.axis('off') # 不要坐標軸
    plt.show()

# Take a batch of data
batch_size_tmp = 8
train_loader_tmp = torch.utils.data.DataLoader(train_ds, batch_size=batch_size_tmp)
dataiter = iter(train_loader_tmp)
images, labels = next(dataiter)
print(images.shape)

# Display image
plt.figure(figsize=(10,6))
imshow(torchvision.utils.make_grid(images))
# Displays classes
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size_tmp)))

def modelTrain(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    loss_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            loss_list.append(loss.item())
            batch = (batch_idx + 1) * len(data)
            data_count = len(train_loader.dataset)
            percentage = (100. * (batch_idx + 1) / len(train_loader))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ' +
                  f'({percentage:.0f} %)  Loss: {loss.item():.6f}')
    return loss_list
def modelTest(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

    # Average loss
    test_loss /= len(test_loader.dataset)
    # Displays the test results
    data_count = len(test_loader.dataset)
    percentage = 100. * correct / data_count
    print(f'Accuracy: {correct}/{data_count} ({percentage:.2f}%)')

epochs = 20
lr=0.05

# Build a model


# Define the loss function
criterion = nn.CrossEntropyLoss()
# Set optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

loss_list = []
for epoch in range(1, epochs + 1):
    loss_list += modelTrain(model, device, train_loader, criterion, optimizer, epoch)
    optimizer.step()

# Plot the loss of the training process
import matplotlib.pyplot as plt

plt.plot(loss_list, 'r')
PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)

model = Net().to(device)
model.load_state_dict(torch.load(PATH))
print(model)
# ---------------------------------------------------------------------------------------------------
# import torch
# import torch.nn as nn
#
# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self.fc1 = nn.Linear(10, 20)
#         self.fc2 = nn.Linear(20, 1)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
# model = SimpleNet().to(device)
# # Print the status dictionary
# print(model.state_dict())
# ---------------------------------------------------------------------------------------------------

modelTest(model, device, test_loader)

batch_size=15
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Display the image
plt.figure(figsize=(10,6))
imshow(torchvision.utils.make_grid(images))

print('Real class: ', ' '.join(f'{classes[labels[j]]:5s}'
                         for j in range(batch_size)))

# predict
outputs = model(images.to(device))

_, predicted = torch.max(outputs, 1)

print('Predict class: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(batch_size)))
# Initialize the correct number for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# predict
batch_size=1000
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predictions = torch.max(outputs, 1)
        # Calculate the correct number for each class
        # zip: mapping the target and the predictions
        for label, prediction in zip(target, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# Calculate the accuracy of each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'{classname:5s}: {accuracy:.1f} %')

