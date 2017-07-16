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
    arg_parse.add_argument("-d", "--dir", default="./grurnn",
                           help="Directory for storing logs.")
    arg_parse.add_argument("-f", "--filename", default="grurnn.log",
                           help="Name of the log file to use.")
    arg_parse.add_argument("-e", "--epochs", default=10, type=int,
                           help="Number of epochs for which to train the RNN.")
    arg_parse.add_argument("-m", "--max", default=None, type=int,
                           help="The maximum number of examples to train on.")
    arg_parse.add_argument("-p", "--patience", default=100000, type=int,
                           help="Number of examples to train before evaluating"
                                + " loss.")
    arg_parse.add_argument("-t", "--test", action="store_true",
                           help="Treat run as test, do not save models")
    arg_parse.add_argument("-l", "--learn_rate", default=0.005, type=float,
                           help="The learning rate to be used in training.")
    arg_parse.add_argument("-o", "--model", default=None,
                           help="Previously trained model to load on init.")
    arg_parse.add_argument("-a", "--anneal", type=float, default=0.00001,
                           help="Sets the minimum possible learning rate.")
    arg_parse.add_argument("-s", "--dataset", default="./datasets/stories.pkl",
                           help="The path to the dataset to be used in "
                                " training.")
    arg_parse.add_argument("-r", "--truncate", type=int, default=100,
                           help="The backpropagate truncate value.")
    arg_parse.add_argument("-i", "--hidden_size", type=int, default=100,
                           help="The size of the hidden layers in the RNN.")
    arg_parse.add_argument("-b", "--embed_size", type=int, default=100,
                           help="The size of the embedding layer in the RNN.")
    return arg_parse.parse_args()
# End of parse_arguments()
