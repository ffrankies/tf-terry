"""
Utility class for setting up an RNN.

Copyright (c) 2017 Frank Derry Wanye

Date: 16 July, 2017
"""

# Specify documentation format
__docformat__ = 'restructedtext en'

import argparse
import logging

def parse_arguments():
    """
    Parses the command line arguments and returns the namespace with those
    arguments.

    :type return: Namespace object
    :param return: The Namespace object containing the values of all supported
                   command-line arguments.
    """
    arg_parse = argparse.ArgumentParser()
    __add_log_arguments__(arg_parse)
    __add_rnn_arguments__(arg_parse)
    __add_train_arguments__(arg_parse)
    return arg_parse.parse_args()
# End of parse_arguments()

def __add_log_arguments__(parser):
    """
    Adds arguments for setting up the logger to the given argument parser.
    Arguments added:
    --logger-name
    --log-filename
    --log-dir

    :type parser: argparse.ArgumentParser
    :param parser: The argument parser to which to add the logger arguments.
    """
    parser.add_argument("-ln", "--logger-name", default="TEST",
                        help="The name of the logger to be used. Defaults to 'TEST'")
    parser.add_argument("-lf", "--log-filename", default="terry.log",
                        help="The name of the file to which the logging will be done.")
    parser.add_argument("-ld", "--log-dir", default="./logging",
                        help="The path to the directory where the log file will be stored.")
# End of __add_log_arguments__()

def __add_rnn_arguments__(parser):
    """
    Adds arguments for setting up an RNN to the given argument parser.
    Arguments added:
    --dataset
    --hidden-size
    --embed-size
    --model

    :type parser: argparse.ArgumentParser
    :param parser: The argument parser to which to add the logger arguments.
    """
    parser.add_argument("-ds", "--dataset", default="./datasets/stories.pkl",
                        help="The path to the dataset to be used for training.")
    parser.add_argument("-hs", "--hidden_size", type=int, default=100,
                        help="The size of the hidden layers in the RNN.")
    parser.add_argument("-es", "--embed_size", type=int, default=100,
                        help="The size of the embedding layer in the RNN.")
    parser.add_argument("-m", "--model", default=None,
                        help="The previously trained model to load on init.")
# End of __add_rnn_arguments__()

def __add_train_arguments__(parser):
    """
    Adds arguments for training an RNN to the given argument parser.
    Arguments added:
    --epochs
    --max
    --patience
    --test
    --learning_rate
    --anneal
    --truncate

    :type parser: argparse.ArgumentParser
    :param parser: The argument parser to which to add the logger arguments.
    """
    parser.add_argument("-e", "--epochs", default=10, type=int,
                        help="The number of epochs for which to train the RNN.")
    parser.add_argument("-max", "--max", default=None, type=int,
                        help="The maximum number of examples to train on.")
    parser.add_argument("-p", "--patience", default=100000, type=int,
                        help="The number of examples to train before evaluating loss.")
    parser.add_argument("-t", "--test", action="store_true",
                        help="Treat run as test, do not save models")
    parser.add_argument("-l", "--learn_rate", default=0.005, type=float,
                        help="The learning rate to be used in training.")
    parser.add_argument("-a", "--anneal", type=float, default=0.00001,
                        help="The minimum possible learning rate.")
    parser.add_argument("-r", "--truncate", type=int, default=100,
                        help="The backpropagate truncate value.")
# End of __add_train_arguments__()

def setup_logger():
    """
    Sets up a logger
    """
    logger = logging.logger()
# End of setup_logger
