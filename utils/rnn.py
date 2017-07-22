"""
An RNN model implementation in tensorflow.

Copyright (c) 2017 Frank Derry Wanye

Date: 19 July, 2017
"""

import numpy
import tensorflow as tf
import logging

from . import constants
from . import setup

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
    # End of __init__()

    def train(self):
        """
        Initial pseudocode for training the model.
        """
        lstm = tf.contrib.rnn.BasicLSTMCell(self.settings.hidden_size)
        batch_size = 1 # My data isn't broken up into batches yet :( Gotta do that, then make this a param.
        state_size = 5 # Size of lstm state... I'm assuming sequence length?
        state = tf.zeros([batch_size, state_size])
        probabilities = []
        loss = 0.0
        # Actual training data in here
    # End of train()
