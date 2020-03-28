# Data ingestion and configuration file ingestion
import configparser

"""Class to hold the configuration file information"""
class Config:
    def __init__(self):


def parse_config(config_file):
    
    config = Config()
    parser = configparser.ConfigParser()
    parser.read(config_file)

    # CNN configurations
    config.cnn_filter_input_sizes = [int(x) for x in parser.get("cnn", "filter_input_sizes").split(",")]
    config.cnn_filter_output_sizes = [int(x) for x in parser.get("cnn", "filter_output_sizes").split(",")]
    config.cnn_kernel_sizes = [int(x) for x in parser.get("cnn", "kernel_sizes").split(",")]
    config.cnn_strides = [int(x) for x in parser.get("cnn", "strides").split(",")]
    config.cnn_pool_kernel_sizes = [int(x) for x in parser.get("cnn", "pool_kernel_sizes").split(",")]

    # Training configurations
    config.data_path = parser.get("training", "data_path")
    config.seed = int(parser.get("training", "seed"))
    config.training_batch_size = int(parser.get("training", "training_batch_size"))
    config.num_epochs = int(parser.get("training", "num_epochs"))
    config.learning_rate = float(parser.get("training", "learning_rate"))
    config.save_path = parser.get("training", "save_path")

    return config


def get_dataset(config):

    return train, valid, test