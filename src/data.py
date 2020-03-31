# Data ingestion and configuration file ingestion
from os import getcwd, listdir
import configparser
import cv2
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pickle


class Config:
    """Class to hold the configuration file information"""
    def __init__(self):
        pass


def parse_config(config_file):
    """Parses the configuration file for information, adds each as
    attribute in a Config class.

    Arguments:
        config_file {string} -- Path to the configuration file

    Returns:
        Config Obj -- Object containing all config information
    """
    config = Config()
    parser = configparser.ConfigParser()
    parser.read(config_file)

    # Dataset configurations
    config.data_path = getcwd() + '/' + parser.get("dataset", "data_path")
    config.image_input_size = tuple([int(i) for i in parser.get("dataset", "image_input_size").split(',')])
    config.validset_seed = int(parser.get("dataset", "validset_seed"))
    config.testset_seed = int(parser.get("dataset", "testset_seed"))
    config.percent_valid = float(parser.get("dataset", "percent_valid"))
    config.percent_test = float(parser.get("dataset", "percent_test"))

    # CNN configurations
    config.filter_input_sizes = [int(x) for x in parser.get("cnn", "filter_input_sizes").split(",")]
    config.filter_output_sizes = [int(x) for x in parser.get("cnn", "filter_output_sizes").split(",")]
    config.kernel_sizes = [int(x) for x in parser.get("cnn", "kernel_sizes").split(",")]
    config.strides = [int(x) for x in parser.get("cnn", "strides").split(",")]
    config.pool_kernel_sizes = [int(x) for x in parser.get("cnn", "pool_kernel_sizes").split(",")]

    # Fully connected configurations

    # Training configurations
    config.training_batch_size = int(parser.get("training", "training_batch_size"))
    config.num_epochs = int(parser.get("training", "num_epochs"))
    config.learning_rate = float(parser.get("training", "learning_rate"))
    config.save_path = parser.get("training", "save_path")

    return config


def get_dataset(config, rebuild=False):
    """Retrieves dataset as formatted from simulated data in Matlab script,
    extracts individual monomer training samples and their exact angle,
    returns as train, test, and validation sets with corresponding labels.

    Arguments:
        config {Config Obj} -- Configuration information

    Keyword Arguments:
        rebuild {bool} -- Decides if dataset should be rebuilt (True) or loaded (default: {False})

    Returns:
        tuple, tuple, tuple -- Train, Valid, and Test sets
                               In each tuple, tuple[0] is images tensors and tuple[1] is labels
    """

    if not rebuild:
        # Load previous build
        dataset = pickle.load(open("dataset_build.p", "rb"))
        return dataset[0], dataset[1], dataset[2]

    # Get filepaths to dataset images
    IMAGE_PATH = config.data_path + 'Samples/'
    image_filenames = [i for i in listdir(IMAGE_PATH) if i[0] != '.']
    # sort filenames
    image_filenames = sorted(image_filenames, key=lambda x: int(x.split('_')[-1][:-4]))

    # Import image data in greyscale
    image_data = []
    for img_file in image_filenames:
        image_data.append(cv2.cvtColor(cv2.imread(IMAGE_PATH+img_file), cv2.COLOR_BGR2GRAY))

    # Import bounding boxes, add index to table
    df_labels = pd.read_csv(config.data_path + 'BoundingBoxes.csv')
    df_labels['x_index'] = df_labels[['Path_Source']].applymap(lambda x: int(x.split('_')[-1][:-4])-1)
    df_labels['theta'] = df_labels['theta'].apply(lambda x: (x + 90) / 180)  # min-max normalize angles to [0,1]

    # Crop each individual monomer into its own image
    monomers = []
    labels = []
    for index, row in df_labels.iterrows():  # for every monomer in labels
        # get bounding box values
        x = row['Y']  # swapped in dataset
        x_end = x+row['w']
        y = row['X']
        y_end = y+row['h']
        # get pixel values
        mnmr = image_data[row['x_index']][x:x_end, y:y_end]
        # zero padding
        padded = np.zeros(config.image_input_size)
        padded[:mnmr.shape[0], :mnmr.shape[1]] = mnmr
        # append data and label
        monomers.append(padded)
        labels.append(row['theta'])
    monomers = torch.Tensor(monomers)
    labels = torch.Tensor(labels)

    # Split into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(monomers, labels,
                                                      test_size=config.percent_test,
                                                      random_state=config.testset_seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp,
                                                          test_size=config.percent_valid/(1-config.percent_test),
                                                          random_state=config.validset_seed)
    # Create sets
    trainset = X_train, y_train
    validset = X_valid, y_valid
    testset = X_test, y_test

    # Save build as binary in data path
    dataset = trainset, validset, testset
    pickle.dump(dataset, open(config.data_path+"dataset_build.p", "wb"))
    
    return trainset, validset, testset
