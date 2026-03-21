import torch
from torch import nn


class CustomNet(nn.Module):
    def __init__(self):
        '''
            Define the CNN model:
            	•	conv1 (kernel = 3, stride = 1, padding = 1) + relu --> input (3, 224, 224) --> output (64, 224, 224)
	            •	maxpool (2,2) --> input (64, 224, 224) --> output (64, 112, 112)
	            •	conv2 + relu --> input (64, 112, 112) --> output (128, 112, 112)
	            •	maxpool --> input (128, 112, 112) --> output (128, 56, 56)
	            •	conv3 + relu --> input (128, 56, 56) --> output (256, 56, 56)
	            •	global avg pool --> input (256, 56, 56) --> output (256, 1, 1)
	            •	linear --> input 256 * 1 * 1 --> output 200
        '''
        super(CustomNet, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # ReLU
        self.relu = nn.ReLU()

        # MaxPooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling: force resolution to be (1,1) without changing the channels
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))

        # final layer
        self.linear1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet   # input (256) --> output (200)




    def forward(self, x):
        # Define forward pass
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.global_avg_pool(x)

        # flatten (256, 1, 1) --> (256)
        x = torch.flatten(x, 1)

        x = self.linear1(x)

        return x