import argparse
from os import getcwd
import data
import models
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from time import time
import pickle


class Trainer():
    def __init__(self, model, config):
        """ Initialize trainer class """
        self.model = model
        self.lr = config.learning_rate
        self.image_size = config.image_input_size
        self.batch_size = config.training_batch_size
        self.epochs = config.num_epochs
        self.save_path = config.save_path
        self.save_path = config.save_path
        self.cur_epoch = 0
        self.validation_frequency = 100  # compute validation loss every Nth training batch
        # Adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # Loss function
        if config.loss_function == 'MSELoss':
            self.loss_func = nn.MSELoss()
        elif config.loss_function == 'MeanCosineLoss':
            self.loss_func = self.model.MeanCosineLoss

    def run_batch(self, batch, targets, train=False):

        if train:
            self.model.train()
            self.model.zero_grad()  # reset gradient
        else:
            self.model.eval()  # deactivate dropout

        # Inference on batch, compute loss
        outputs = self.model(batch)
        loss = self.loss_func(outputs, targets)

        if train:
            loss.backward()  # compute gradient
            self.optimizer.step()  # change model weights
        return loss

    def train(self, config, X_train, y_train, X_valid, y_valid):

        # Train
        train_history = []  # training loss
        valid_history = []  # validation loss
        for epoch in tqdm(range(self.epochs)):
            for i in range(0, len(X_train), self.batch_size):
                # Get one batch from data
                X_batch = X_train[i:i+self.batch_size].view((-1, 1)+self.image_size)
                y_batch = y_train[i:i+self.batch_size].view((-1, 1))
                # Run batch
                loss = self.run_batch(X_batch, y_batch, train=True)

                # Append (batch number, loss value)
                train_history.append((i/self.batch_size, loss))
                # Test on validation set
                if i % (self.validation_frequency * self.batch_size) == 0:
                    val_loss = self.run_batch(X_valid.view((-1, 1)+self.image_size), y_valid.view(-1, 1))

                    valid_history.append((i/self.batch_size, val_loss))

        return train_history, valid_history

    def plot_loss(self, config, train_history, valid_history):
        # Training Loss
        batch_num_train = [i[0] for i in train_history]
        loss_train = [i[1] for i in train_history]
        plt.plot(batch_num_train, loss_train, color='blue', label='Training')
        # Validation Loss
        batch_num_valid = [i[0] for i in valid_history]
        loss_valid = [i[1] for i in valid_history]
        plt.plot(batch_num_valid, loss_valid, color='orange', label='Validation')
        # Labels
        plt.title('Training and Validation Loss')
        plt.xlabel('Batch Number')
        plt.ylabel(f'Loss ({config.loss_function})')
        plt.show()


if __name__ == "__main__":

    # Parse arguments for training script
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true', help='rebuild dataset')
    parser.add_argument('--plot', action='store_true', help='output plots')
    parser.add_argument('--save', action='store_true', help='save trained model')
    args = parser.parse_args()
    rebuild_flag = args.rebuild  # if dataset should be rebuilt or loaded from pickle file
    plot_flag = args.plot  # if plots from loss function should be output
    save_flag = args.save  # if model should be saved as file

    # Get configs and dataset
    PATH = getcwd() + '/src/'
    config = data.parse_config(PATH + 'config.cfg')
    trainset, validset, testset = data.get_dataset(config, rebuild=rebuild_flag)
    # Initialize model
    cnn = models.Net(config)
    # Grab training and validation data
    X_train = trainset[0]
    y_train = trainset[1]
    X_valid = validset[0]
    y_valid = validset[1]

    # Run training script
    trainer = Trainer(cnn, config)
    train_history, valid_history = trainer.train(config, X_train, y_train, X_valid, y_valid)

    # Save model
    if save_flag:
        version = str(int(time()))
        model_name = 'model-'+version+'.p'
        pickle.dump(cnn, open(PATH + config.model_path + model_name, "wb"))
        config_name = 'model-'+version+'_configs.p'
        pickle.dump(config, open(PATH + config.model_path + config_name, "wb"))

    # Plot loss
    if plot_flag:
        trainer.plot_loss(config, train_history, valid_history)
