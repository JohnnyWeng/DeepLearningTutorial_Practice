import os
import torch
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt

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
for i in range(1, cols * rows + 1):
    # .item() 取出值
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
print(img.shape)

# Get the specific image
import matplotlib.pyplot as plt
import skimage

orig_img = skimage.data.astronaut()
# You can save the image
skimage.io.imsave('images_test/astronaut.jpg', orig_img)
plt.imshow(orig_img)

print(skimage.data.__all__)
