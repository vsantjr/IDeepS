
'''
Author: Valdivino Alexandre de Santiago Júnior

This program was developed based on an official PyTorch tutorial (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) created by Nathan Inkawhich. It is a program addressing the deep convolutional generative adversarial network DCGAN. 

If you want to see the notebook version of this code with detailed explanations, take a look at: https://github.com/vsantjr/DeepLearningMadeEasy/blob/temp_23-09/PyTorch_DCGAN.ipynb
'''

'''
Before running the code, do the following:

1.) Download the test.zip (https://github.com/vsantjr/IDeepS/blob/master/Datasets/test.zip) dataset which is a very small subset (1,294 images) of the imagenettetvt320 (https://www.kaggle.com/datasets/valdivinosantiago/imagenettetvt320) dataset;

2.) Create a directory in your working directory called "img";

3.) Unzip the dataset into the "img" directory: unzip test.zip -d ./img. 
'''

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
from IPython.display import HTML
from prettytable import PrettyTable
from PIL import Image

# Set random seed for reproducibility.
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results.
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# This function obtains the number of trainable parameters of the model/network.
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total trainable params: {total_params}")
    return total_params



# All relevant outputs are here. Create directory if it does not exist.
if not os.path.exists('outputs'):
    os.makedirs('outputs')
    
output_dir = './outputs'


# Root directory for dataset.
dataroot = "img"

# Number of workers for dataloader.
workers = 2

# Batch size during training.
batch_size = 128

# Spatial size of training images. All images will be resized to this
# size using a transformer.
image_size = 64

# Number of channels in the training images. 
nc = 3

# Size of z latent vector (i.e. size of generator input).
nz = 100

# Size of feature maps in generator.
ngf = 64

# Size of feature maps in discriminator.
ndf = 64

# Number of training epochs.
num_epochs = 20 

# Learning rate for optimizers.
lr = 0.0002

# Beta1 hyperparam for Adam optimizers.
beta1 = 0.5

# Configuration:
#    - use 0 for CPU mode (no GPU); 
#    - use 1 for single node/single GPU.
ngpu = 1
if (ngpu == 0):
    print('Configuration: CPU (no GPU)!')
elif (ngpu == 1):    
    print('Configuration: Single Node/Single GPU!')
else:
    print('Invalid Configuration!')


# Load data
dataset = dset.ImageFolder(root=dataroot,
                          transform=transforms.Compose([
                          transforms.Resize(image_size),
                          transforms.CenterCrop(image_size),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
# Create the dataloader.
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on.
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Check the number of available GPUs.
print ('Available GPUs: ', torch.cuda.device_count())

# Save some training images.
real_batch = next(iter(dataloader))
for i in range(len(real_batch)):
  print('Image: ', i, ' - Input shape: ', real_batch[i].shape )

plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig(f"{output_dir}/some_training_images.png")



# Custom weights initialisation called on netG and netD.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# The Generator.
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)



# Create the generator.
netG = Generator(ngpu).to(device)

# Apply the weights_init function. 
netG.apply(weights_init)

# Check number of trainable parameters.
print('Checking trainable parameters: {}'.format(count_parameters(netG)))

# Print the model.
print(netG)

# The Discriminator.
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
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
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



# Create the Discriminator.
netD = Discriminator(ngpu).to(device)
    
# Apply the weights_init function.
netD.apply(weights_init)

# Check number of trainable parameters.
print('Checking trainable parameters: {}'.format(count_parameters(netD)))

# Print the model.
print(netD)


# Initialise BCELoss function.
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualise the progression of the generator.
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training.
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D.
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# Function to check the average values of tensors.
def meanTensor(t):
  meant = torch.sum(t)
  mean = meant.cpu().detach().numpy()
  mean = mean / t.size()
  return mean


# Training Loop.
# Lists to keep track of progress.
img_list = []
G_losses = []
D_losses = []
iters = 0
meanDx, meanDGz1, meanDGz2 = 0.0, 0.0, 0.0

print("Starting Training Loop...")
# For each epoch.
for epoch in range(num_epochs):
    # For each batch in the dataloader.
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch.
        netD.zero_grad()
        # Format batch.
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D.
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch.
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass.
        errD_real.backward()
        D_x = output.mean().item()
        meanDx = meanTensor(output) # To be sure.

        ## Train with all-fake batch.
        # Generate batch of latent vectors.
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G.
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D.
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch.
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients.
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        meanDGz1 = meanTensor(output) # To be sure.
        # Compute error of D as sum over the fake and the real batches.
        errD = errD_real + errD_fake
        # Update D.
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # Fake labels are real for generator cost.
        # Since we just updated D, perform another forward pass of all-fake batch through D.
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output.
        errG = criterion(output, label)
        # Calculate gradients for G.
        errG.backward()
        D_G_z2 = output.mean().item()
        meanDGz2 = meanTensor(output) # To be sure.
        # Update G.
        optimizerG.step()

        # Output training stats.
        if i % 50 == 0:
             print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f -- %.4f\tD(G(z1 / z2)): %.4f -- %.4f / %.4f -- %.4f'
                    % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, meanDx, D_G_z1, meanDGz1, D_G_z2, meanDGz2))    
        
        # Save losses for plotting later.
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise.
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1



plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"{output_dir}/g_d_loss.png")

# Grab a batch of real images from the dataloader.
real_batch = next(iter(dataloader))

# Save the real images.
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.savefig(f"{output_dir}/real_images.png")

# Save the fake images from the last epoch.
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig(f"{output_dir}/fake_images.png")
