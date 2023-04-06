import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function

#Test
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # input 32x32 
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        # Layer:
        self.conv1 = nn.Conv2d(1, 6, 5)         #Input -> C1 -> S2  “activation map”  in: 1, out: 6, kernel: 5
        self.conv2 = nn.Conv2d(6, 16, 5)        #S2-C3-S4 downsampled “activation map” in layer S2
       
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # S4*5*5 -> F5 from full connected
        self.fc2 = nn.Linear(120, 84)           # featuresF5 -> F6  fully connected
        self.fc3 = nn.Linear(84, 10)            # F6-> Output 10 Gaussian

    #Comutation:
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))                 #Input -> C1 -> S2
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)                      #S2-C3-S4   
        x = x.view(-1, self.num_flat_features(x))                       # 
        x = F.relu(self.fc1(x))                                         # S4*5*5 -> F5 linear ReLu
        x = F.relu(self.fc2(x))                                         # F5 -> F6 linear ReLu
        x = self.fc3(x)                                                 # F6-> Output Gaussian
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = LeNet()
print(net)                         # what does the object tell us about itself?

input = torch.rand(1, 1, 32, 32)   # stand-in for a 32x32 black & white image
print('\nImage batch shape:')
print(input.shape)

output = net(input)                # we don't call forward() directly
print('\nRaw output:')
#print(output)
print(output.shape)

#%matplotlib inline

import torch
import torchvision
import torchvision.transforms as transforms

transformer = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transformer)

#When we instantiate our dataset, we need to tell it a few things:
# - The filesystem path to where we want the data to go.
# - Whether or not we are using this set for training; most datasets will be split into training and test subsets.
# - Whether we would like to download the dataset if we haven’t already.
# - The transformations we want to apply to the data.

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

import matplotlib.pyplot as plt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
