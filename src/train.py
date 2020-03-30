from os import getcwd
import data
import models
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


def mean_angular_loss(prediction, target):
    angular_dist = torch.abs(prediction - target)
    losses = torch.where((90 < angular_dist) & (angular_dist <= 180), 180 - angular_dist, angular_dist)
    return torch.mean(losses**2)


def train(config, X_train, y_train, X_valid, y_valid):
    # Configurations
    batch_size = config.training_batch_size
    # Initialize model
    cnn = models.Net(config)
    # Set up optimizer
    optimizer = optim.Adam(cnn.parameters(), lr=config.learning_rate)
    #loss_function = mean_angular_loss(outputs, y_batch)
    loss_function = nn.MSELoss()
    
    # Train
    history = []
    for epoch in tqdm(range(config.num_epochs)):
        cnn.train()
        for i in range(0, len(X_train), batch_size):
            # Get one batch from data
            X_batch = X_train[i:i+batch_size].view((-1, 1)+config.image_input_size)
            y_batch = y_train[i:i+batch_size]
            # Reset gradient
            cnn.zero_grad()
            # Run model on batch, get loss
            outputs = cnn(X_batch)
            loss = loss_function(outputs, y_batch.view(-1, 1))
            # Compute gradient from loss
            loss.backward()
            # Take a step with optimizer
            optimizer.step()

        # Test on validation set
        cnn.eval()
        history.append(test(cnn, X_valid.view((-1, 1)+config.image_input_size), y_valid))
    return history


def test(model, X_test, targets):
    preds = model(X_test)
    return mean_angular_loss(preds, targets)


if __name__ == "__main__":

    # Get configs and dataset
    PATH = getcwd() + '/'
    config = data.parse_config(PATH + 'config.cfg')
    trainset, validset, testset = data.get_dataset(config)
    # Grab training and validation data
    X_train = trainset[0]
    y_train = trainset[1]
    X_valid = validset[0]
    y_valid = validset[1]
    # Run training script
    history = train(config, X_train, y_train, X_valid, y_valid)

    plt.plot(history)
    plt.show()
