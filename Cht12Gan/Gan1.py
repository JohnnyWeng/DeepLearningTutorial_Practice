import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Set the hyperparameters
latent_dim = 1
data_dim = 2
batch_size = 32
epochs = 10000
lr = 0.001

# Generate real-world data: (x, y) pairs
def real_data_sampler(num_samples):
    x = np.linspace(-1, 1, num_samples)
    y = x**2
    data = np.vstack((x, y)).T
    return data

import numpy as np

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# Stacked vertically
stacked = np.vstack((x, y))
print('stacked = ', stacked)
# [[1 2 3]
#   [4 5 6]]

# 反轉上面的 stacked: horizontal -> vertical
transposed = stacked.T
print('transposed = ', transposed)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, data_dim)
        )
    def forward(self, z):
        return self.model(z)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, data):
        return self.model(data)

def trainingModel():
    # Training Loop: Train the GAN model
    for epoch in range(epochs):
        optimizer_D.zero_grad()
        # Real and Fake Data Preparation
        # Real data
        real_data = torch.Tensor(real_data_sampler(batch_size))
        # valid "label": 1
        valid = torch.ones(batch_size, 1)
        # fake "label": 0
        fake = torch.zeros(batch_size, 1)
        print('valid = ', valid) # matrix of 1
        print('fake = ', fake) # matrix of 0
        '''real loss'''
        # real data is the parabola (red line)
        # we want to let the model knows that the real data is valid
        real_loss = adversarial_loss(discriminator(real_data), valid)
        # Fake data: random  data
        z = torch.randn(batch_size, latent_dim)
        generated_data = generator(z)
        # Compare data with fake data. If we get 1 and compare with fake(0) data, we will get loss.
        '''fake loss'''
        # we want to let the model knows that the fake data is fake
        fake_loss = adversarial_loss(discriminator(generated_data), fake)
        d_loss = (real_loss + fake_loss) / 2

        # Backpropagation and Discriminator Update
        # We train the model with the true and fake data loss combined.
        d_loss.backward()
        optimizer_D.step()  # Update the parameters of discriminator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        generated_data = generator(z)
        g_loss = adversarial_loss(discriminator(generated_data), valid)
        g_loss.backward()
        optimizer_G.step()  # Update the parameters of generator
        # Outputs training progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

def plot():
    # Plot the resulting data points
    z = torch.randn(100, latent_dim)
    generated_data = generator(z).detach().numpy()
    real_data = real_data_sampler(100)
    # real data
    plt.scatter(real_data[:, 0], real_data[:, 1], color='red', label='Real data')
    # generated data
    plt.scatter(generated_data[:, 0], generated_data[:, 1], color='blue', label='Generated data')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Models (Generator & Discriminator)
    generator = Generator()
    discriminator = Discriminator()
    # Loss function: Binary Cross-Entropy Loss (二元分類)
    adversarial_loss = nn.BCELoss()

    # Optimizer
    # generator is the model that we want to train
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    # training
    trainingModel()

    # plot
    plot()