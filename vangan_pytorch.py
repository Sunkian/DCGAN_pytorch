from __future__ import print_function


"""
DCGAN Implementation
*Author: `Alice Pagnoux <https://github.com/Sunkian> `
*Title: Deep Convolutional Generative Adversarial Network (DCGAN)
*Description: This code has been written in Python for my Msc. Dissertaion Project and represents the implementation
of a DCGAN.
"""

"""
Parameters I need to change and compare the result:
    - the dataset size (better results with bigger datasets)
    - the batch size (64 or 128)
    - the optimizers (Adam, SGD, RMSprop) for G and D
    - the number of epochs
    - the learning rate lr
"""

# About GANs

# -----------------------------------------------------------------------------------------------------------------

# Originally proposed by Ian Goodfellow in 2014, Generative Adversarial Networks (GANs) are a part of the
# Deep Generative model family. The main purpose of this algorithm results in the confrontation of two neural
# networks to generate new kind of data. Getting a random noise as an input, the Generator tries to create data
# and to fool the Discriminator. This second neural networks gets, on one hand, the dataset with the real images,
# then the 'generated images' from G. It tried to distinguish the real data from the fake ones. Combined with a
# backpropagation system that helps both networks to improve themselves, the algorithm learns to generate new data
# with the same characteristics as the training set.

# -----------------------------------------------------------------------------------------------------------------

# Imports

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters

# ---------
#
#  - *dataroot* - Directory for the dataset.
#  - *batch_size* - Number of samples that will be propagated through the network. Here we have 4319 files,
#  #so if the batch size = 128, the algorithm will take the first 128 samples to train the network, then takes the
#  #next 128 samples and train the network again.
#  - *img_size* - We will resize all the images to this size so they have the same spatial characteristics.
#  - *nb_channels* - The number of channels. For colored images, it's 3.
#  - *latent_space* - Length of the noise vector (input for the Generator)
#  - *generator_output_size* - The size of the image output by the generator.
#  - *discriminator_input_size* - The size of the image input to the discriminator.
#  - *nb_epochs* - The number of training epochs.
#  - *lr* - Learning rate for the optimizers (Adam). The value refers to Ian Goodfellow"s paper.
#  - *beta1* - Hyperparameter for Adam optimizer. The value refers to Ian Goodfellow"s paper.
#  - *nb_gpu* - Number of gpus available. Use 0 for CPU mode. (Pytorch documentation).

dataroot = "./data/1"
batch_size = 128
img_size = 64
nb_channels = 3
latent_space = 100
gos = 64
dis = 64
nb_epochs = 2
lr = 0.0002
beta1 = 0.5
nb_gpu = 1

# ---------

# Reformat and load the data

# -------------------------

# Compose is an equivalent to 'map' in JS (Check https://pytorch.org/docs/stable/torchvision/transforms.html)
# We resize the data so they all have the same spacial characteristics
dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

# Creation of the data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and nb_gpu > 0) else "cpu")

# Plot some images from the training dataset
# for i in dataloader:
#     real_batch = next(iter(i)) #Image by image
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
# 0: 1st axis, 1: 2nd axis, 2: 3rd axis 1 2 0 is to have the images displayed in the right position. If we do 2 1 0 they
# will be flipped on the left side for example
plt.show()


# -------------------------

# Weights initialization

# ---------------------

# As specified in Ian Goodfellow's paper, all the model weights shall be randomly initialized from a
# Normal distribution with mean=0, stdev=0.02. The weights_init function takes as an input a model and reinitializes
# All Convo and BatchNorm layers to meet the following initial criteria.

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


# ---------------------

# Generator

# ---------

class Generator(nn.Module):
    def __init__(self, nb_gpu):
        super(Generator, self).__init__()
        self.nb_gpu = nb_gpu
        self.main = nn.Sequential(
            # The generator takes a [100] vector noise as an input, going into a 512 convolution
            nn.ConvTranspose2d(latent_space, gos * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gos * 8),
            nn.ReLU(True),

            # 512 x 4 x 4
            nn.ConvTranspose2d(gos * 8, gos * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gos * 4),
            nn.ReLU(True),

            # 256 x 8 x 8
            nn.ConvTranspose2d(gos * 4, gos * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gos * 2),
            nn.ReLU(True),

            # 128 x 16 x 16
            nn.ConvTranspose2d(gos * 2, gos, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gos),
            nn.ReLU(True),

            # 64 x 32 x 32
            nn.ConvTranspose2d(gos, nb_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Now we are at 3 x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# ---------

# # --------------------------------------------------------------------------------------------------

# Create the Generator, Handle the weights and the gpu related to it and print the model architecture
Gnetwork = Generator(nb_gpu).to(device)

# Handle mutli gpu if desired
if (device.type == 'cuda') and (nb_gpu > 1):
    Gnetwork = nn.DataParallel(Gnetwork, list(range(nb_gpu)))

# Randomly initialize all the weights to mean = 0 and stdev = 0.2.
Gnetwork.apply(weights_init)

# Print the model architecture
print(Gnetwork)


# # --------------------------------------------------------------------------------------------------


# Discriminator

# -------------

class Discriminator(nn.Module):
    def __init__(self, nb_gpu):
        super(Discriminator, self).__init__()
        self.nb_gpu = nb_gpu
        self.main = nn.Sequential(
            # The input of the Discriminator is the number of channels x 64 x 64 = 3 x 64 x 64
            nn.Conv2d(nb_channels, dis, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            # 64 x 32 x 32
            nn.Conv2d(dis, dis * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dis * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            # 128 x 16 x 16
            nn.Conv2d(dis * 2, dis * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dis * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            # 256 x 8 x 8
            nn.Conv2d(dis * 4, dis * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dis * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            # 512 x 4 x 4
            nn.Conv2d(dis * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# -------------

# -------------------------------------------------------------------------------------------------------

# Create the discriminator, Handle the weights and the gpu related to it and print the model architecture

Dnetwork = Discriminator(nb_gpu).to(device)

if (device.type == 'cuda') and (nb_gpu > 1):
    Dnetwork = nn.DataParallel(Dnetwork, list(range(nb_gpu)))

Dnetwork.apply(weights_init)

print(Dnetwork)

# -------------------------------------------------------------------------------------------------------

# Loss functions and Optimizers

# -----------------------------

# Initialization of the Binary Cross Entropy loss function (BCE)
# Creates a criterion that measures the Binary Cross Entropy between the target and the output (
# ref https://pytorch.org/docs/master/generated/torch.nn.BCELoss.html)
loss = nn.BCELoss()

# Real labels = 1, Fake labels = 0

# D and G have the same optimizer; the Adam optimizer with the same learning rate = 0.0002 and beta1 = 0.5.
# As advised in the Pytorch tutorial, we will generate a fixed batch of latent vectors that are drawn form a
# Gaussian  distribution to keep track of the generator's learning progression.

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_n = torch.randn(64, latent_space, 1, 1, device=device)

real_results_label = 1
fake_results_label = 0

# Set up Adam optimizer
# TRY TO USE SGD (stochastic gradient descent) or RMSprop ?
Goptimizer = optim.Adam(Gnetwork.parameters(), lr=lr, betas=(beta1, 0.999))
Doptimizer = optim.Adam(Dnetwork.parameters(), lr=lr, betas=(beta1, 0.999))

# -----------------------------

# Training (based on Algorithm 1 on Goodfellow's paper)

# -----------------------------------------------------

img_list = []
G_losses = []
D_losses = []
iterations = 0

print("Starting the Training...")

for epoch in range(nb_epochs):  # We do it all the epochs
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        Dnetwork.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_results_label, device=device)
        # Forward pass real batch through D
        output = Dnetwork(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = loss(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, latent_space, 1, 1, device=device)
        # Generate fake image batch with G
        fake = Gnetwork(noise)
        label.fill_(fake_results_label)
        # Classify all fake batch with D
        output = Dnetwork(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = loss(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        Doptimizer.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        Gnetwork.zero_grad()
        label.fill_(real_results_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = Dnetwork(fake).view(-1)
        # Calculate G's loss based on this output
        errG = loss(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        Goptimizer.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, nb_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iterations % 500 == 0) or ((epoch == nb_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = Gnetwork(fixed_n).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iterations += 1

# -----------------------------------------------------

# Results

# -------

# Plot (graph) of D & G’s losses versus training iterations.
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# **Real Images vs. Fake Images**
#
# Finally, lets take a look at some real images and fake images side by
# side.
#

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()

# -------
