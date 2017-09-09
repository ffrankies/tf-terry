"""
Utility class for setting up an RNN.

Copyright (c) 2017 Frank Derry Wanye

Date: 09 August, 2017
"""

# Specify documentation format
__docformat__ = 'restructedtext en'

import argparse
import logging
import logging.handlers
import os
import sys

from . import constants

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
    __add_dataset_arguments__(arg_parse)
    return arg_parse.parse_args()
# End of parse_arguments()

def __add_log_arguments__(parser):
    """
    Adds arguments for setting up the logger to the given argument parser.
    Arguments added:
    --logger_name
    --log_filename
    --log_dir

    :type parser: argparse.ArgumentParser
    :param parser: The argument parser to which to add the logger arguments.
    """
    parser.add_argument("-ln", "--logger_name", default=constants.LOG_NAME,
                        help="The name of the logger to be used. Defaults to %s" % constants.LOG_NAME)
    parser.add_argument("-lf", "--log_filename", default="terry.log",
                        help="The name of the file to which the logging will be done.")
    parser.add_argument("-ld", "--log_dir", default="./logging/",
                        help="The path to the directory where the log file will be stored.")
# End of __add_log_arguments__()

def __add_rnn_arguments__(parser):
    """
    Adds arguments for setting up an RNN to the given argument parser.
    Arguments added:
    --dataset
    --hidden_size
    --embed_size
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
    parser.add_argument("-m", "--model_name", default=None,
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
    --batch_size

    :type parser: argparse.ArgumentParser
    :param parser: The argument parser to which to add the logger arguments.
    """
    parser.add_argument("-e", "--epochs", default=10, type=int,
                        help="The number of epochs for which to train the RNN.")
    parser.add_argument("-max", "--max", default=None, type=int,
                        help="The maximum number of examples to train on.")
    parser.add_argument("-p", "--patience", default=100000, type=int,
                        help="The number of examples to train before evaluating loss.")
    parser.add_argument("-test", "--test", action="store_true",
                        help="Treat run as test, do not save models")
    parser.add_argument("-l", "--learn_rate", default=0.005, type=float,
                        help="The learning rate to be used in training.")
    parser.add_argument("-a", "--anneal", type=float, default=0.00001,
                        help="The minimum possible learning rate.")
    parser.add_argument("-t", "--truncate", type=int, default=10,
                        help="The backpropagate truncate value.")
    parser.add_argument("-b", "--batch_size", type=int, default=5,
                        help="The size of the batches into which to split the training data.")
# End of __add_train_arguments__()

def __add_dataset_arguments__(parser):
    """
    Adds arguments for training an RNN to the given argument parser.
    Arguments added:
    --raw_data_path
    --saved_dataset_path
    --saved_dataset_name
    --source_type

    :type parser: argparse.ArgumentParser
    :param parser: The argument parser to which to add the logger arguments.
    """
    parser.add_argument("-rdp", "--raw_data_path", default="./raw_data/stories.csv",
                        help="The path to the existing data.")
    parser.add_argument("-sdp", "--saved_dataset_path", default="./datasets",
                        help="The directory for the saved dataset.")
    parser.add_argument("-sdn", "--saved_dataset_name", default="test.pkl",
                        help="The name of the saved dataset.")
    parser.add_argument("-st", "--source_type", default="csv",
                        help="The type of source data [currently only the csv data size is supported].")
# End of __add_dataset_arguments__()

def create_dir(dirPath):
    """
    Creates a directory if it does not exist.

    :type dirPath: string
    :param dirPath: the path of the directory to be created.
    """
    try:
        if os.path.dirname(dirPath) != "":
            os.makedirs(os.path.dirname(dirPath), exist_ok=True) # Python 3.2+
    except TypeError:
        try: # Python 3.2-
            os.makedirs(dirPath)
        except OSError as exception:
            if exception.errno != 17:
                raise
# End of create_dir()

def get_arg(settings, argument, asInt=False, asBoolean=False, asFloat=False, checkNone=False):
    """
    Retrieves the argument from the given settings namespace, or asks the user to enter one.

    :type settings: argparse.Namespace() object
    :param settings: The parsed command-line arguments to the program.

    :type argument: String
    :param argument: the argument to retrieve from the settings.

    :type return: Any
    :param return: The value of the returned setting.
    """
    arg = None
    if argument in settings:
        arg = vars(settings)[argument]
    if checkNone:
        if arg == None:
            arg = input("Specify the value for %s" % argument)
    if asInt:
        arg = int(arg)
    if asBoolean:
        if arg.lower() == 'true' or arg.lower() == 't':
            arg = True
        if arg.lower() == 'false' or arg.lower() == 'f':
            arg = False
    if asFloat:
        arg = float(arg)
    return arg
# End of get_arg()

def setup_logger(args):
    """
    Sets up a logger

    :type args: Namespace object.
    :param args: The namespace containing command-line arguments entered (or their default values).
    """
    logger = logging.getLogger(args.logger_name) if args.model_name is None else logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)

    create_dir(args.log_dir)
    logger_filename = args.log_filename if args.model_name is None else args.model_name + ".log"

    # Logger will use up to 5 files for logging, 'rotating' the data between them as they get filled up.
    handler = logging.handlers.RotatingFileHandler(
        filename=args.log_dir + '/' + logger_filename,
        maxBytes=1024*512,
        backupCount=5
    )

    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.info("Logger successfully set up.")
    return logger
# End of setup_logger
