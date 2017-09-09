"""
An RNN model implementation in tensorflow.

Copyright (c) 2017 Frank Derry Wanye

Date: 9 September, 2017
"""

import numpy as np
import tensorflow as tf
import logging
import ray

from . import constants
from . import setup
from . import datasets
from . import saver

class RNNModel(object):
    """
    A basic RNN implementation in tensorflow.
    """

    def __init__(self, args=None, saved_model_path=None):
        """
        Constructor for an RNN.

        :type args: Namespace object.
        :param args: The namespace of the command-line arguments passed into the class, or their default values. If
                     not provided, the RNN will initialize those params to defaults.
        """
        self.settings = setup.parse_arguments() if args is None else args
        self.logger = setup.setup_logger(self.settings)
        self.logger.info("RNN settings: %s" % self.settings)
        self.__create_ops__()
    # End of __init__()

    def __create_ops__(self):
        """
        Creates all internal tensorflow operations and variables inside a local graph and session.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.__load_dataset__()
            self.__unstack_variables__()
            self.__create_functions__()
            self.session = tf.Session(graph=self.graph)
            self.variables = ray.experimental.TensorFlowVariables(self.total_loss_fun, self.session)
            self.session.run(tf.global_variables_initializer())
        self.model_path = saver.create_model_dir(self.settings)
    # End of __create_ops__()

    def __pad_2d_matrix__(self, matrix, value=None):
        """
        Pads the rows of a 2d matrix with either a given value or the last value in each 
        row.

        :type matrix: nested list
        :param matrix: 2d matrix in python list form with variable row length.

        :type value: int
        :param value: the value to append to each row.

        :type return: nested list
        :param return: 2d matrix in python list form with a fixed row length.
        """
        self.logger.debug("Padding matrix with shape: ", matrix.shape)
        paddedMatrix = matrix
        maxRowLength = max([len(row) for row in paddedMatrix])
        for row in paddedMatrix:
            while len(row) < maxRowLength:
                row.append(value) if value is not None else row.append(row[-1])
        return paddedMatrix
    # End of __pad_2d_matrix__()

    def __list_to_numpy_array__(self, matrix):
        """
        Converts a list of list matrix to a numpy array of arrays.

        :type matrix: nested list
        :param matrix: 2d matrix in python list form.

        :type return: nested numpy array
        :param return: the matrix as a numpy array or arrays.
        """
        paddedMatrix = self.__pad_2d_matrix__(matrix, value=None)
        return np.array([np.array(row, dtype=int) for row in paddedMatrix], dtype=int)
    # End of __list_to_numpy_array__()

    def __2d_list_to_long_array__(self, matrix):
        array = np.array([])
        for row in matrix: array = np.append(array, row)
        return array
    # End of __2d_list_to_long_array__()

    def __load_dataset__(self):
        """
        Loads the dataset specified in the command-line arguments. Instantiates variables for the class.
        """
        dataset_params = datasets.load_dataset(self.logger, self.settings)
        # Don't need to keep the actual training data when creating batches.
        self.vocabulary, self.index_to_word, self.word_to_index, x_train, y_train = dataset_params
        self.index_to_word = dataset_params[1]
        self.word_to_index = dataset_params[2]
        x_train = self.__2d_list_to_long_array__(x_train)
        y_train = self.__2d_list_to_long_array__(y_train)
        self.__create_batches__(x_train, y_train)
    # End of load_dataset()

    def __create_batches__(self, x_train, y_train):
        """
        Creates batches out of loaded data.

        Current implementation is very limited. It would probably be best to sort the training data based on length, 
        fill it up with placeholders so the sizes are standardized, and then break it up into batches.
        """
        self.logger.info("Breaking input data into batches.")
        self.x_train_batches = x_train.reshape((self.settings.batch_size,-1))
        self.y_train_batches = y_train.reshape((self.settings.batch_size,-1))
        # self.x_train_batches = np.array([ x_train[i:i+self.settings.truncate] for i in range(len(x_train)-self.settings.truncate+1) ])
        # self.y_train_batches = np.array([ y_train[i:i+self.settings.truncate] for i in range(len(y_train)-self.settings.truncate+1) ])
        self.num_batches = len(self.x_train_batches)
        self.logger.info("Obtained %d batches." % self.num_batches)
    # End of __create_batches__() 

    def __create_variables__(self):
        """
        Creates placeholders and variables for the tensorflow graph.
        """
        self.batch_x_placeholder = tf.placeholder(
            tf.int32, 
            [self.settings.batch_size, self.settings.truncate],
            name="input_placeholder")
        self.batch_y_placeholder = tf.placeholder(
            tf.float32,
            np.shape(self.batch_x_placeholder),
            name="output_placeholder")
        self.hidden_state_placeholder = tf.placeholder(
            tf.float32, 
            [self.settings.batch_size, self.settings.hidden_size],
            name="hidden_state_placeholder")

        self.vocabulary_size = len(self.index_to_word)
        self.out_weights = tf.Variable(
            np.random.rand(self.settings.hidden_size, self.vocabulary_size), 
            dtype=tf.float32,
            name="out_weights")
        self.out_bias = tf.Variable(
            np.zeros((1, self.vocabulary_size)), 
            dtype=tf.float32,
            name="out_bias")

        self.embeddings = tf.get_variable(
            name="word_embeddings",
            shape=[self.vocabulary_size, self.settings.hidden_size],
            dtype=tf.float32
        )
    # End of __create_placeholders__()

    def __unstack_variables__(self):
        """
        Splits tensorflow graph into adjacent time-steps.
        """
        self.__create_variables__()
        # The embedding lookup simulates one-hot encoding of the input,
        # and simplifies the vectors being used.
        inputs = tf.nn.embedding_lookup(
            params=self.embeddings, ids=self.batch_x_placeholder,
            name="embedding_lookup")

        self.inputs_series = tf.unstack(
            inputs, axis=1, 
            name="unstack_inputs_series")
        self.outputs_series = tf.unstack(
            self.batch_y_placeholder, axis=1, 
            name="unstack_outputs_series")
    # End of __unpack_variables__()

    def __forward_pass__(self):
        """
        Performs a forward pass within the RNN.

        :type return: tuple consisting of two matrices (list of lists)
        :param return: (states_series, current_state)
        """
        cell = tf.contrib.rnn.GRUCell(self.settings.hidden_size, reuse=tf.get_variable_scope().reuse)
        states_series, current_state = tf.contrib.rnn.static_rnn(cell, self.inputs_series, self.hidden_state_placeholder)
        return states_series, current_state
    # End of __forward_pass__()

    def __create_functions__(self):
        """
        Creates symbolic functions needed for training.

        Functions created: predictions_series - makes output predictions on a forward pass
                           total_loss_fun - calculates the loss for a forward pass
                           train_step_fun - performs back-propagation for the forward pass
        """
        states_series, self.current_state = self.__forward_pass__()
        # logits_series.shape = (truncate, num_batches, vocabulary_size)
        logits_series = [
            tf.matmul(state, self.out_weights, name="state_times_out_weights") + self.out_bias 
            for state in states_series] #Broadcasted addition
        self.predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
        # Logits = predictions before softmax
        # Predictions_series = softmax(logits) (make probabilities add up to 1)

        losses = []
        for logits, labels in zip(logits_series, self.outputs_series):
            labels = tf.to_int32(labels, "CastLabelsToInt")
            losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        self.total_loss_fun = tf.reduce_mean(losses)
        self.train_step_fun = tf.train.AdagradOptimizer(self.settings.learn_rate).minimize(self.total_loss_fun)
    # End of __create_functions__()
