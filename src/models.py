import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 32, 2)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.conv3 = nn.Conv2d(64, 128, 2)

        # Check output size from convolution layers
        self._num_conv_features = None
        x = torch.randn(11, 11).view(-1, 1, 11, 11)  # random input
        self.convs(x)  # single forward pass through convs

        # Fully connected layers
        self.fc1 = nn.Linear(self._num_conv_features, 75)
        self.fc2 = nn.Linear(75, 30)
        self.fc3 = nn.Linear(30, 13)

    def convs(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Determine flattened output size from convolutions
        if self._num_conv_features is None:
            self._num_conv_features = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)  # run through convolution layers
        x = x.view(-1, self._num_conv_features)  # flatten
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # linear to get regression result
        x = self.fc3(x)
        return F.softmax(x, dim=1)
