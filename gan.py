from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Root directory for dataset
dataroot = "./data"

# Number of workers for dataloader
workers = 5

# Batch size during training
batch_size = 10

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 32

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 10

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 1

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

def imgshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

transform = transforms.Compose([
                               transforms.Resize(image_size),
                            #    transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) 
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=10)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) # descargar el dataset de entrenamiento
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True, num_workers=2) # carga el dataset de entrenamiento




# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code

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
            # nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf*2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Net(nn.Module):
    def __init__(self, ngpu):
        super(Net, self).__init__()
        self.ngpu = ngpu
        # self.main = nn.Sequential(
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # )
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
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
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Create the generator
netG = Generator(ngpu).to(device)

cnn = Net(ngpu)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

print("Starting CNN Training Loop")
for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
print('Finished CNN Training')

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)

# Create the Discriminator
netD = Net(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)

# Initialize BCELoss function
criterion = nn.CrossEntropyLoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.SGD(netD.parameters(), lr=0.001, momentum=0.9)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
fakeList = []
G_losses = []
D_losses = []
iters = 0

print("Starting GAN Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        inputs, labels = data

        # zero the parameter gradients
        optimizerD.zero_grad()

        # forward + backward + optimize
        outputs = netD(inputs)
        errD_real = criterion(outputs, labels)
        errD_real.backward()
        optimizerD.step()
        D_x = outputs.mean().item() #sacar?

        ## Train with all-fake batch
        # Generate batch of latent vectors
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        labels.fill_(fake_label)
        labels = labels.to(device=device, dtype=torch.int64)

        # Classify all fake batch with D
        output = netD(fake.detach())
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, labels)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labels.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake)
        # Calculate G's loss based on this output
        errG = criterion(output, labels)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                fakeList.append(fake) # se debe llamar a fakeList[-1] para obtener la ultimas imagenes generadas
                # for pic in fake:  #duda, aca se agregan las fotos por separados ? <- asumire que si by: NFS
                #     fakeList.append(pic)
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
        
# HTML(ani.to_jshtml())
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#assign labels to the fake image generate
fakeLabels = []
fakeList = fakeList[-1] #this is the last set of picture generate (the best)
outputs = cnn(fakeList) 
_, predicted = torch.max(outputs, 1)

for i in range(0,len(fakeList)-1):
    fakeLabels.append(predicted[i])

#add the pictures generated by the gan to the trainset  
for j in range(0,len(fakeList)-1): #check the range of iteration! 
        new_data= torch.tensor((np.transpose(np.array(fakeList[j]).reshape(1, 3, 32, 32), (0,2 ,3 ,1)))) 
        dataset.data = np.r_[dataloader.dataset.data, new_data] 

dataset.targets.append(fakeLabels)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=10) #convert the new dataset in a tensor (?)

#where are the trainset ? -> it is the dataloader
#I put a testset to set the cnn_2

# Now must to be the code to train the cnn with the new dataset. --v

# uncomment this if it is ok.

# cnn_2 = Net()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(cnn_2.parameters(), lr=0.001, momentum=0.9)

# for epoch in range(7):  # loop over the dataset multiple times
#   running_loss = 0.0
#   for i, data in enumerate(dataloader, 0):
#       # get the inputs; data is a list of [inputs, labels]
#       inputs, labels = data


#       # zero the parameter gradients
#       optimizer.zero_grad()

#       # forward + backward + optimize
#       outputs = cnn2(inputs)
#       loss = criterion(outputs, labels)
#       loss.backward()
#       optimizer.step()

#       # print statistics
#       running_loss += loss.item()
#       if i % 2000 == 1999:    # print every 2000 mini-batches
#           print('[%d, %5d] loss: %.3f' %
#                 (epoch + 1, i + 1, running_loss / 2000))
#           running_loss = 0.0

# print('Finished Training')

# correct = 0
# total = 0
# with torch.no_grad():
#   for data in testloader: #we are testing the predictions with the original data. is  it okay ? or we have to add the new data to the testset too ? if it that the case we must to do again the before step
#       images, labels = data
#       outputs = cnn_2(images)
#       _, predicted = torch.max(outputs.data, 1)
#       total += labels.size(0)
#       correct += (predicted == labels).sum().item()

# print('Accuracy of the network with the new dataset : %d %%' % (100 * correct / total))


######

#muchas dudas aca, lo cambiare segun mi pinta.

#images = img_list[-1]
# fakeLabels = []
# for elem in fakeList:
#     output = cnn(fakeList[0])
#     _, predicted = torch.max(outputs, 1)
#     fakeLabels.append(predicted)

# outputs = cnn(images)
# _, predicted = torch.max(outputs, 1)

# plt.imshow(np.transpose(fakeList[-1],(1,2,0)))

# print('Predicted:', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
# imgshow(torchvision.utils.make_grid(images))