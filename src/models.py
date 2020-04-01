import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        # Convolution layers
        self.convs = []
        for i in range(self.config.num_conv_layers):
            self.convs.append(nn.Conv2d(self.config.cnn_features[i],
                                        self.config.cnn_features[i+1],
                                        self.config.kernel_sizes[i],
                                        self.config.cnn_stride))
        # Pooling layer
        if self.config.pool_layer:
            self.pool = nn.MaxPool2d(self.config.pool_size, self.config.pool_stride)

        # Check output size from convolution layers
        self._num_conv_features = None
        x = torch.randn(config.image_input_size).view((-1, 1)+config.image_input_size)  # random input
        self.convolutions(x)  # single forward pass through convs
        self.config.fc_features = [self._num_conv_features] + self.config.fc_features  # define input size to fcs

        # Fully connected layers
        self.fcs = []
        self.drops = []
        for i in range(self.config.num_fc_layers):
            self.fcs.append(nn.Linear(self.config.fc_features[i],
                                      self.config.fc_features[i+1]))
            self.drops.append(nn.Dropout(self.config.dropout_probs[i]))
        self.linear = nn.Linear(self.config.fc_features[-1], 1)

    def MeanCosineLoss(self, outputs, targets):
        # Convert normalized angle into radians
        outputs = (outputs-0.5) / math.pi
        targets = (targets-0.5) / math.pi
        # Compute absolute angular differences
        angular_diff = torch.abs(outputs - targets)
        # Compute cosine loss
        loss = torch.abs(torch.cos(angular_diff))
        return torch.mean(loss)  # return mean loss

    def convolutions(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))

        if self.config.pool_layer:
            x = self.pool(x)

        # Determine flattened output size from convolutions
        if self._num_conv_features is None:
            self._num_conv_features = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def fully_connecteds(self, x):
        for fc, drop in zip(self.fcs, self.drops):
            x = F.relu(fc(x))
            x = drop(x)
        return x

    def forward(self, x):
        # Run through convolution layers
        x = self.convolutions(x)
        x = x.view(-1, self._num_conv_features)  # flatten
        # Fully connected layers
        x = self.fully_connecteds(x)

        x = self.linear(x)  # linear to get regression result
        return x
