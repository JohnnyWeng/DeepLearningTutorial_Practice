import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
from torch.utils.data import DataLoader
import os
from torchvision.models import ResNet18_Weights

os.environ["KMP_DUPLICATE_LIB_OK"] = "True "
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the pre-trained ResNet-18 model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

for name, param in model.named_parameters():
    if 'fc' not in name:  # Only train last layer fc
        print('name = ', name)
        param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 2- categories classification task
print('model = ', model)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Mean, Fit ImageNet, Standard Deviation
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# Load training and test datasets
train_data = datasets.ImageFolder('dog_vs_cat/train', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_data = datasets.ImageFolder('dog_vs_cat/test', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    # 32 data each batch
    for images, labels in train_loader:
        # Move data to the appropriate device (e.g., GPU if available)
        images, labels = images.to(device), labels.to(device)
        model.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels) # calculate the loss (y-y')

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')
print('Training complete')

model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # Disable gradient calculation for efficiency and testing.
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        # Outputs: Probability of the two classes.
        outputs = model(images)
        print('outputs =', outputs)
        _, predicted = torch.max(outputs.data, 1) # The 1st dimension
        print('predicted =', predicted) # predicted: 0 or 1
        # The total of data.
        total += labels.size(0)
        # how many true?
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total} %')

for name, param in model.named_parameters():
    if 'fc' not in name:  # Only train last layer fc
        print('name = ', name)
        param.requires_grad = False
from torchinfo import summary
print(summary(model, (1, 3, 224, 224)))

