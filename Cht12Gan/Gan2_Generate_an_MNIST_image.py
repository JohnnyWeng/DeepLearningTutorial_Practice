import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set parameters
batch_size = 64
learning_rate = 0.0002
num_epochs = 100
latent_dim = 100

# Dataset (using MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

# Define generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            # Adds non-linearity, allowing small negative gradients when inputs are less than zero (helps prevent dead neurons).
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.view(-1, 28 * 28))

# Initialize generator and discriminator, and move them to the device
# These are our models.
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Set loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Function to display generated images
def show_generated_images(epoch, generator, num_images=4):
    generator.eval()  # Set generator to evaluation mode
    with torch.no_grad():  # No need for gradient computation
        fig, axes = plt.subplots(2, 2, figsize=(5, 5))
        for i, ax in enumerate(axes.flatten()):
            # Generate a different random noise for each image
            z = torch.randn(1, latent_dim).to(device)
            generated_image = generator(z).cpu()  # Generate image and move to CPU for display
            ax.imshow(generated_image.squeeze(), cmap='gray')
            ax.axis('off')
    plt.tight_layout()
    plt.show()


for epoch in range(num_epochs):
    # Loop all the data in the dataloader.
    for i, (images, _) in enumerate(data_loader):
        batch_size = images.size(0)
        images = images.to(device)
        # real, fake labels
        # 1
        real_labels = torch.ones(batch_size, 1).to(device) # real data from MINIST.
        # 0
        fake_labels = torch.zeros(batch_size, 1).to(device)
        optimizer_D.zero_grad()
        # input images to the model and outputs the answers.
        # images: a batch of images
        outputs = discriminator(images)
        # real data needs to distinguished as real data, while fake data needs to distinguished as fake data
        d_loss_real = criterion(outputs, real_labels)
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        # We hope that the fake data generated will be 0.
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device) # Generate Noise
        # Generate fake images
        fake_images = generator(z)
        print('fake_images length = ', len(fake_images)) #64
        outputs = discriminator(fake_images)
        # Each value in the tensor represents the discriminator's confidence that the corresponding fake image is real (i.e., it belongs to the real MNIST dataset).
        print('outputs = ', outputs)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()
        # Generate some images at the end of each epoch
    show_generated_images(epoch, generator)
    print(f'Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

