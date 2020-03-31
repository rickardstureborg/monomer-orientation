import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 32, 2)
        self.conv2 = nn.Conv2d(32, 128, 2)
        self.pool = nn.MaxPool2d(2, 2)

        # Check output size from convolution layers
        self._num_conv_features = None
        x = torch.randn(config.image_input_size).view((-1, 1)+config.image_input_size)  # random input
        self.convs(x)  # single forward pass through convs

        # Fully connected layers
        self.fc1 = nn.Linear(self._num_conv_features, 200)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(200, 50)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, 1)

    def MeanCosineLoss(self, outputs, targets):
        # Convert normalized angle into radians
        outputs = (outputs-0.5) / math.pi
        targets = (targets-0.5) / math.pi
        # Compute absolute angular differences
        angular_diff = torch.abs(outputs - targets)
        # Compute cosine loss
        loss = torch.abs(torch.cos(angular_diff))
        return torch.mean(loss)  # return mean loss

    def convs(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.pool(x))

        # Determine flattened output size from convolutions
        if self._num_conv_features is None:
            self._num_conv_features = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        # Run through convolution layers
        x = self.convs(x)
        x = x.view(-1, self._num_conv_features)  # flatten

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))

        x = self.fc3(x)  # linear to get regression result
        return x
