import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_data = FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

labels_map = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

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
train_loader = DataLoader(training_data, batch_size=600)
# Set optimizer
optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
model.train() # set the model to training mode
loss_list = []

for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Usually, we need to input the 28 * 28
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
test_loader = DataLoader(test_data, shuffle=False, batch_size=test_data.targets.shape[0])

model.eval()
test_loss = 0
correct = 0
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    output = model(data)

    # sum up batch loss
    test_loss += criterion(output, target).item()
    output_cpu = output.argmax(dim=1).cpu().numpy()
    target_cpu = target.cpu().numpy()
    print('Confusion Matrix = \n', confusion_matrix(output_cpu, target_cpu))

    # output: The output of the prediction. argmax: get the highest prediction
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
# normalize
test_loss /= len(test_loader.dataset)
batch = batch_idx * len(data)
data_count = len(test_loader.dataset)
percentage = 100. * correct / data_count
# 5 -> 6 (wrong)
print(f'correct_rate: {correct}/{data_count}' + f' ({percentage:.0f}%)\n') # 91% -> Every 10 images will have a wrong one.


for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
print(img.shape)

import matplotlib.pyplot as plt
import skimage
orig_img = skimage.data.astronaut()
skimage.io.imsave('images_test/astronaut.jpg', orig_img)
plt.imshow(orig_img)

print(skimage.data.__all__)
