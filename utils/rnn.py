"""
An RNN model implementation in tensorflow.

Copyright (c) 2017 Frank Derry Wanye

Date: 23 July, 2017
"""

import numpy
import tensorflow as tf
import logging

from . import constants
from . import setup
from . import datasets

class RNN(object):
    """
    A basic RNN implementation in tensorflow.
    """

    def __init__(self, args=None):
        """
        Constructor for an RNN.

        :type args: Namespace object.
        :param args: The namespace of the command-line arguments passed into the class, or their default values. If
                     not provided, the RNN will initialize those params to defaults.
        """
        if args is None:
            self.settings = setup.parse_arguments()
            self.logger = setup.setup_logger(self.settings)
        else:
            self.settings = args
            self.logger = setup.setup_logger(self.settings)
        # End of else
        self.logger.info("RNN settings: %s" % self.settings)
        self.__load_dataset__()
        self.bias = tf.Variable(tf.zeros([3, setup.get_arg(self.settings, 'hidden_size')]))
        self.out_bias = tf.Variable(tf.zeros([len(self.vocabulary)]))
    # End of __init__()

    def __load_dataset__(self):
        """
        Loads the dataset specified in the command-line arguments. Instantiates variables for the class.
        """
        dataset_params = datasets.load_dataset(self.logger, self.settings)
        self.vocabulary = dataset_params[0]
        self.index_to_word = dataset_params[1]
        self.word_to_index = dataset_params[2]
        self.x_train = dataset_params[3]
        self.y_train = dataset_params[4]
    # End of load_dataset()

    def train(self):
        """
        Initial pseudocode for training the model.
        """
        lstm = tf.contrib.rnn.BasicLSTMCell(self.settings.hidden_size)
        batch_size = 1 # My data isn't broken up into batches yet :( Gotta do that, then make this a param.
        state_size = 5 # Size of lstm state... I'm assuming sequence length?
        state = tf.zeros([batch_size, state_size])
        probabilities = []
        dataset = []
        loss = 0.0
        # Actual training data in here
        for batch in dataset:
            output, state = lstm(batch, state)
            logits = tf.matmul(output, softmax_w) + softmax_b
            loss += loss_function(probabilities, target_words)
    # End of train()

    def generate_output(self):
        print("This feaure isn't implemented yet!")
    # End of generate_output()
