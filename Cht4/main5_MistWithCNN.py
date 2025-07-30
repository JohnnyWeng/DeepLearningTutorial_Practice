import torch
from torch import nn
from torch.nn import functional as F
# from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

torch.manual_seed(16)
PATH_DATASETS = ""
BATCH_SIZE = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download training data of MNIST
train_ds = MNIST(PATH_DATASETS, train=True, download=True,
                 transform=transforms.ToTensor())
# Download test data of MNIST
test_ds = MNIST(PATH_DATASETS, train=False, download=True,
                 transform=transforms.ToTensor())
print(train_ds.data.shape, test_ds.data.shape)
print(train_ds.targets.shape, test_ds.targets.shape) #target


# Define CNN Model:
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        '''
        We have two layers.
        '''
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # This line flattens the tensor from a 2D (or higher) shape into a 1D vector per sample, while keeping the batch size (out.size(0)) intact.
        out = out.reshape(out.size(0), -1)  # explain later
        # full connected
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        return out

model = ConvNet().to(device)
print(model)

# Training Setup:
epochs = 10
lr=0.1
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Initialize Training:
model.train()
loss_list = []

# Training Loop:
for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        #  In this case, it's used for "multi-class classification", where the output from the neural network is typically log-probabilities (produced by log_softmax). The NLLLoss function compares the predicted log-probabilities with the actual class labels and gives a value indicating how far off the predictions are.
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Monitor and Log Training Progress:
        if (batch_idx + 1) % 10 == 0:
            loss_list.append(loss.item())
            batch = (batch_idx + 1) * len(data)
            data_count = len(train_loader.dataset)
            percentage = (100. * (batch_idx + 1) / len(train_loader))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)' +
                  f'  Loss: {loss.item():.6f}')
# Plotting Training Loss:
import matplotlib.pyplot as plt
plt.plot(loss_list, 'r')
plt.show()

# Create DataLoader
test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE)

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target).item()
        output = model(data)
        # count correctness
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == target).sum().item()

# Average loss
test_loss /= len(test_loader.dataset)
# Show test result
batch = batch_idx * len(data)
data_count = len(test_loader.dataset)
percentage = 100. * correct / data_count
print(f'Average loss: {test_loss:.4f}, Correctness rate: {correct}/{data_count}' +
      f' ({percentage:.2f}%)\n')

# Predict 20 test data
predictions = []
with torch.no_grad():
    for i in range(20):
        data, target = test_ds[i][0], test_ds[i][1]
        data = data.reshape(1, *data.shape).to(device)
        output = torch.argmax(model(data), axis=-1)
        predictions.append(str(output.item()))

# matching
print('actual    :', test_ds.targets[0:20].numpy())
print('prediction: ', ' '.join(predictions[0:20]))

# Show the probability of the 9th datum
import numpy as np

i=18
data = test_ds[i][0]
data = data.reshape(1, *data.shape).to(device)
print(data.shape)
predictions = torch.softmax(model(data), dim=1)
print(f' 0~9 predict probability : {np.around(predictions.cpu().detach().numpy(), 2)}')
print(f'0~9 predict result : {np.argmax(predictions.cpu().detach().numpy(), axis=-1)}')

X2 = test_ds[i][0]
plt.imshow(X2.reshape(28,28), cmap='gray')
plt.axis('off')
plt.show()

# Save model
torch.save(model, 'cnn_model.pth')

# Load model
model = torch.load('cnn_model.pth')

from skimage import io
from skimage.transform import resize

no=9
uploaded_file = f'../Cht3/myDigits/{no}.png'
image1 = io.imread(uploaded_file, as_gray=True)

data_shape = data.shape
image_resized = resize(image1, data_shape[2:], anti_aliasing=True)
X1 = image_resized.reshape(*data_shape) #/ 255.0
# Invert the color, in real handwritten, background is white.
X1 = 1.0-X1

import matplotlib.pyplot as plt

plt.imshow(X1.reshape(28,28), cmap='gray')
plt.axis('off')
plt.show()

data_shape = X1.shape
print(data_shape)

X1 = torch.FloatTensor(X1).to(device)

# predict
predictions = model(X1)
print(f'actual/prediction: {no} {np.argmax(predictions.detach().cpu().numpy())}')

print(model(X1))

# Read the image and turn to monochrome
for i in range(10):
    uploaded_file = f'../Cht3/myDigits/{no}.png'
    image1 = io.imread(uploaded_file, as_gray=True)

    #  Image scaled to (28, 28) size
    image_resized = resize(image1, tuple(data_shape)[2:], anti_aliasing=True)
    X1 = image_resized.reshape(*data_shape)

    # Invert the color, color 0 is white, unlike RGB color coding, its 0 is black
    X1 = 1.0 - X1
    X1 = torch.FloatTensor(X1).to(device)

    # predict
    predictions = torch.softmax(model(X1), dim=1)
    print(f'actual/prediction: {i} {np.argmax(predictions.detach().cpu().numpy())}')

# Displays summary information for the model
for name, module in model.named_children():
    print(f'{name}: {module}')
