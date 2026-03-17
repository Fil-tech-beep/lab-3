import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)       # input (3, 224, 224) --> output (64, 224, 224)
                                                                      # then --> ReLU
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)     # input (64, 224, 224) --> output (128, 224, 224)
                                                                      # then --> ReLU
                                                                      # then Maxpooling: (128, 224, 224) --> (128, 112, 112)
        # ReLu
        self.relu = nn.ReLU()

        # MaxPooling --> MAH, serve davvero???
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolution block 2: (128, 112, 112) --> (256, 112, 112)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)    # input (128, 112, 112) --> output (256, 112, 112)
                                                                      # then --> ReLU
                                                                      # then --> Global Average Pooling: (256, 112, 112) --> (256, 1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))

        # final layer
        self.linear1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet   # input (256) --> output (200)





    def forward(self, x):
        # Define forward pass
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.Maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.global_avg_pool(x)

        # flatten (256, 1, 1) --> (256)
        x = torch.flatten(x, 1)


        x = self.linear1(x)

        return x