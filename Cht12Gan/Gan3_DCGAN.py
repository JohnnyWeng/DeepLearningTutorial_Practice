import os
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision import utils as vutils
torch.manual_seed(16)
PATH_DATASETS = "" # Default path
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["KMP_DUPLICATE_LIB_OK"]= "True"
# 轉換
transform=transforms.Compose([
   transforms.Resize(28),
   transforms.ToTensor(),
   transforms.Normalize((0.5,), (0.5,)),
])
# Loaded with MNIST handwritten training data
dataset = MNIST(PATH_DATASETS, train=True, download=True,
                 transform=transform)
dataloader = torch.utils.data.DataLoader(dataset
                   , batch_size=BATCH_SIZE, shuffle=True)

# The dimensions of the training data
print(dataset.data.shape)

nz = 100  # The noise dimension of GENERATOR
ngf = 64  # The filter numbers of GENERATOR
ndf = 64  # The filter numbers of DISCRIMINATOR

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02) # The initial weight value of the convolutional layer
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02) # The initial weight value of Batch Normalization
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1,
                               stride=1, padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output
netG = Generator().to(device)
print(netG)

from torchinfo import summary
print(summary(netG))
print()

class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

netD = Discriminator().to(device)

print(netD)

print(summary(netG))
print()
# Set the loss function
criterion = nn.BCELoss()

# Set optimizer
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1.0
fake_label = 0.0
niter = 25
# Model training
for epoch in range(niter):
    for i, data in enumerate(dataloader, 0):
        ########################################################
        # (1) Discriminant neural networks: maximize log(D(x)) + log(1 - D(G(z)))
        #######################################################
        # Train real data
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = netD(real_cpu)
        #print("netD_output",output.shape)  #netD_output torch.Size([64])
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Training fake information
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        # print("netG_fake", fake.shape) # netG_fake torch.Size([64, 1, 28, 28])
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ########################################################
        # (2) Discriminant neural networks: maximize log(D(G(z)))
        #######################################################
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        if i % 200 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch + 1, niter, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            vutils.save_image(real_cpu, 'gan_output/real_samples.png', normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(), 'gan_output/fake_samples_epoch_%03d.png'
                              % (epoch), normalize=True)
    torch.save(netG.state_dict(), 'netG_epoch_%d.pth' % (epoch))
    torch.save(netD.state_dict(), 'gan_weights/netD_epoch_%d.pth' % (epoch))

import matplotlib.pyplot as plt

batch_size = 25
latent_size = 100

fixed_noise = torch.randn(batch_size, latent_size, 1, 1).to(device)
fake_images = netG(fixed_noise)
fake_images_np = fake_images.cpu().detach().numpy()
fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 28, 28)
R, C = 5, 5
for i in range(batch_size):
    plt.subplot(R, C, i + 1)
    plt.axis('off')
    plt.imshow(fake_images_np[i], cmap='gray')
plt.show()

import imageio
import glob

# Produce a GIF file
anim_file = './gan_output/dcgan.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('./gan_output/fake_samples*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

