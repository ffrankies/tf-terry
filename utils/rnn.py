"""
An RNN model implementation in tensorflow.

Copyright (c) 2017 Frank Derry Wanye

Date: 23 August, 2017
"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
        self.settings = setup.parse_arguments() if args is None else args
        self.logger = setup.setup_logger(self.settings)
        self.logger.info("RNN settings: %s" % self.settings)
        self.__load_dataset__()
        self.bias = tf.Variable(tf.zeros([3, setup.get_arg(self.settings, 'hidden_size')]))
        self.out_bias = tf.Variable(tf.zeros([len(self.vocabulary)]))
    # End of __init__()

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
        paddedMatrix = self.__pad_2d_matrix__(matrix, value=len(self.index_to_word))
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
        self.x_train_batches = np.asarray(x_train).reshape((self.settings.batch_size,-1))
        self.y_train_batches = np.asarray(y_train).reshape((self.settings.batch_size,-1))
        self.num_batches = len(self.x_train_batches)
        self.logger.info("Obtained %d batches." % self.num_batches)
    # End of __create_batches__() 

    def __create_variables__(self):
        """
        Creates placeholders and variables for the tensorflow graph.
        """
        self.batch_x_placeholder = self.batch_y_placeholder = tf.placeholder(tf.float32, [self.settings.batch_size, self.settings.truncate])
        self.hidden_state_placeholder = tf.placeholder(tf.float32, [self.settings.batch_size, self.settings.hidden_size])

        vocabulary_size = len(self.index_to_word) + 1
        self.W2 = tf.Variable(np.random.rand(self.settings.hidden_size, vocabulary_size), dtype=tf.float32)
        self.b2 = tf.Variable(np.zeros((1, vocabulary_size)), dtype=tf.float32)
    # End of __create_placeholders__()

    def __unstack_variables__(self):
        """
        Splits tensorflow graph into adjacent time-steps.
        """
        self.__create_variables__()
        self.inputs_series = tf.split(value=self.batch_x_placeholder, num_or_size_splits=self.settings.truncate, axis=1)
        self.outputs_series = tf.unstack(self.batch_y_placeholder, axis=1)
    # End of __unpack_variables__()

    def __forward_pass__(self):
        """
        Performs a forward pass within the RNN.

        :type return: tuple consisting of two matrices (list of lists)
        :param return: (states_series, current_state)
        """
        cell = tf.contrib.rnn.BasicRNNCell(self.settings.hidden_size)
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
        logits_series = [tf.matmul(state, self.W2) + self.b2 for state in states_series] #Broadcasted addition
        self.predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

        losses = []
        for logits, labels in zip(logits_series, self.outputs_series):
            labels = tf.to_int32(labels, "CastLabelsToInt")
            losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        self.total_loss_fun = tf.reduce_mean(losses)
        self.train_step_fun = tf.train.AdagradOptimizer(self.settings.learn_rate).minimize(self.total_loss_fun)
    # End of __create_functions__()

    def __plot__(self, loss_list):
        """
        Plots a graph of epochs against losses.
        Currently, the loss_list stores data by minibatch, so it has to be traversed by
        skipping num_batches steps.

        :type loss_list: list()
        :param loss_list: the losses incurred during training.
        """
        epoch_loss_list = loss_list[::self.num_batches]
        plt.plot(range(1, len(epoch_loss_list) + 1), epoch_loss_list)
        plt.savefig(self.settings.log_dir + "plot.png")
        plt.show()
    # End of plot()

    def __train_minibatch__(self, batch_num, sess, current_state):
        """
        Trains one minibatch.

        :type batch_num: int
        :param batch_num: the current batch number.

        :type sess: tensorflow Session
        :param sess: the session during which training occurs.

        :type current_state: numpy matrix (array of arrays)
        :param current_state: the current hidden state

        :type return: (float, numpy matrix)
        :param return: (the calculated loss for this minibatch, the updated hidden state)
        """
        start_index = batch_num * self.settings.truncate
        end_index = start_index + self.settings.truncate

        batch_x = self.x_train_batches[:, start_index:end_index]
        batch_y = self.y_train_batches[:, start_index:end_index]
        total_loss, train_step, current_state, predictions_series = sess.run(
            [self.total_loss_fun, self.train_step_fun, self.current_state, self.predictions_series],
            feed_dict={
                self.batch_x_placeholder:batch_x, 
                self.batch_y_placeholder:batch_y, 
                self.hidden_state_placeholder:current_state
            })
        return total_loss, current_state, predictions_series
    # End of __train_minibatch__()

    def __train_epoch__(self, epoch_num, sess, current_state, loss_list):
        """
        Trains one full epoch.

        :type epoch_num: int
        :param epoch_num: the number of the current epoch.

        :type sess: tensorflow Session
        :param sess: the session during training occurs.

        :type current_state: numpy matrix
        :param current_state: the current hidden state.

        :type loss_list: list of floats
        :param loss_list: holds the losses incurred during training.

        :type return: (float, numpy matrix)
        :param return: (the latest incurred lost, the latest hidden state)
        """
        self.logger.info("Starting epoch: %d" % (epoch_num))

        for batch_num in range(self.num_batches):
            # Debug log outside of function to reduce number of arguments.
            self.logger.debug("Training minibatch : ", batch_num, " | ", "epoch : ", epoch_num + 1)
            total_loss, current_state, predictions_series = self.__train_minibatch__(batch_num, sess, current_state)
            loss_list.append(total_loss)
        # End of batch training

        self.logger.info("Finished epoch: %d | loss: %f" % (epoch_num, total_loss))
        return total_loss, current_state, predictions_series
    # End of __train_epoch__()

    def train(self):
        """
        Trains the given model on the given dataset, and saves the losses incurred
        at the end of each epoch to a plot image.
        """
        self.logger.info("Started training the model.")
        self.__unstack_variables__()
        self.__create_functions__()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loss_list = []

            current_state = np.zeros((self.settings.batch_size, self.settings.hidden_size), dtype=float)
            for epoch_idx in range(1, self.settings.epochs + 1):
                total_loss, current_state, predictions_series = self.__train_epoch__(epoch_idx, sess, current_state, loss_list)
                # End of epoch training

        self.logger.info("Finished training the model. Final loss: %f" % total_loss)
        self.__plot__(loss_list)
        self.generate_output()
    # End of train()

    def __sample__(self):
        """
        Sample RNN output.
        """
        return tf.multinomial(self.logits, 1)
    # End of __sample__()

    def generate_output(self):
        """
        Generates output sentences.
        """
        self.logger.info("Generating output.")
        sentence = np.array(self.word_to_index[constants.SENT_START])
        print("This feaure isn't implemented yet!")
    # End of generate_output()

    def save_output(self):
        """
        Saves sentence output to a file.
        """
        print("This feature isn't implemented yet!")
    # End of save_output()
