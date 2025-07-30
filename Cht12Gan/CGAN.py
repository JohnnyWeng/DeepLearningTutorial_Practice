import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST

BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5,), (0.5,)),
])
dataset = FashionMNIST('', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(dataset[0][0])


class generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Sets the embedding layer as input to Label
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # 64 data at a time: 64's 100
        z = z.view(z.size(0), 100)  # torch.Size([64, 100]) -> 100 dimensions
        c = self.label_emb(labels)  # torch.Size([64, 10]) -> 10 classes
        x = torch.cat([z, c], 1)  # Merge inputs torch.Size([64, 110]) -> 10+100
        out = self.model(x) # 784 dimensions from the above Generator class (28*28)
        # convert and return the 28*28
        return out.view(x.size(0), 28, 28)
class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Sets the embedding layer as input to Label
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x, labels):
        x = x.view(x.size(0), 784) #torch.Size([64, 784])
        c = self.label_emb(labels) #torch.Size([64, 10])
        # 1: real or not in boolean format.
        x = torch.cat([x, c], 1)# Merge inputs torch.Size([64, 794])
        out = self.model(x)
        return out.squeeze()
# Instantiate the models
discriminator = discriminator().to(device)
generator = generator().to(device)
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).to(device)
    # randomly create 0~10 classes
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).to(device)# Random random numbers [1, 10]
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data.item()

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()
    # Train for real image
    # real data turns out to abe real, which is all we want.
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))
    # Train for fake image
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).to(device)#Random random numbers [1, 10]
    # fake labels will be related to the fake images
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    # For the fake_validity, it should be close to the '0' gradually.
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data.item()
num_epochs = 30
n_critic = 5
display_step = 300
for epoch in range(num_epochs):
    print('Starting epoch {}...'.format(epoch))
    for i, (images, labels) in enumerate(data_loader):
        real_images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        generator.train()
        batch_size = real_images.size(0)
        d_loss = discriminator_train_step(len(real_images), discriminator, generator, d_optimizer, criterion, real_images, labels)
        g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)
    generator.eval()
    print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))
    z = Variable(torch.randn(9, 100)).to(device)
    labels = Variable(torch.LongTensor(np.arange(9))).to(device)
    sample_images = generator(z, labels).unsqueeze(1).data.cpu()
    grid = make_grid(sample_images, nrow=3, normalize=True).permute(1,2,0).numpy()
    plt.imshow(grid)
    plt.axis('off')
    plt.show()
