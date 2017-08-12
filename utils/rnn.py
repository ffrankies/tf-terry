"""
An RNN model implementation in tensorflow.

Copyright (c) 2017 Frank Derry Wanye

Date: 11 August, 2017
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
        self.__create_batches__()
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

    def __load_dataset__(self):
        """
        Loads the dataset specified in the command-line arguments. Instantiates variables for the class.
        """
        dataset_params = datasets.load_dataset(self.logger, self.settings)
        self.vocabulary = dataset_params[0]
        self.index_to_word = dataset_params[1]
        self.word_to_index = dataset_params[2]
        self.x_train = self.__list_to_numpy_array__(dataset_params[3])
        self.y_train = self.__list_to_numpy_array__(dataset_params[4])
    # End of load_dataset()

    def __create_batches__(self):
        """
        Creates batches out of loaded data.

        Current implementation is very limited. It would probably be best to sort the training data based on length, 
        fill it up with placeholders so the sizes are standardized, and then break it up into batches.
        """
        self.logger.info("Breaking input data into batches.")
        self.x_train_batches = np.asarray(self.x_train).reshape((self.settings.batch_size,-1))
        self.y_train_batches = np.asarray(self.y_train).reshape((self.settings.batch_size,-1))
        # self.x_train_batches = [self.x_train[i:i+self.settings.batch_size] 
        #     for i in range(0, len(self.x_train), self.settings.batch_size)]
        # self.y_train_batches = [self.y_train[i:i+self.settings.batch_size] 
        #     for i in range(0, len(self.y_train), self.settings.batch_size)]
        # maxBatchLength = 0
        # for batch in self.x_train_batches:
        #     batchLength = len(batch)
        #     maxBatchLength = batchLength if batchLength > maxBatchLength else maxBatchLength
        # placeholder = len(self.vocabulary)
        # self.x_train_batches = np.pad(self.x_train_batches, maxBatchLength, 'edge')
        # self.y_train_batches = np.pad(self.y_train_batches, maxBatchLength, 'edge')
        # self.x_train_batches = (self.x_train_batches + [placeholder] * maxBatchLength)[:maxBatchLength]
        # self.y_train_batches = (self.y_train_batches + [placeholder] * maxBatchLength)[:maxBatchLength]
        self.num_batches = len(self.x_train_batches)
        self.logger.info("Obtained %d batches." % self.num_batches)
    # End of __create_batches__() 

    def __create_variables__(self):
        """
        Creates placeholders and variables for the tensorflow graph.
        """
        self.batch_x_placeholder = self.batch_y_placeholder = tf.placeholder(tf.float32, [self.settings.batch_size, self.settings.truncate])
        self.hidden_state = tf.placeholder(tf.float32, [self.settings.batch_size, self.settings.hidden_size])

        self.W = tf.Variable(np.random.rand(self.settings.hidden_size+1, self.settings.hidden_size), dtype=tf.float32)
        self.b = tf.Variable(np.zeros((1,self.settings.hidden_size)), dtype=tf.float32)

        vocabulary_size = len(self.index_to_word) + 1
        self.W2 = tf.Variable(np.random.rand(self.settings.hidden_size, vocabulary_size), dtype=tf.float32)
        self.b2 = tf.Variable(np.zeros((1, vocabulary_size)), dtype=tf.float32)
    # End of __create_placeholders__()

    def __unstack_variables__(self):
        """
        Splits tensorflow graph into adjacent time-steps.
        """
        self.__create_variables__()
        self.inputs_series = tf.unstack(self.batch_x_placeholder, axis=1)
        self.outputs_series = tf.unstack(self.batch_y_placeholder, axis=1)
    # End of __unpack_variables__()

    def __forward_pass__(self):
        """
        Performs a forward pass within the RNN.

        :type return: a Matrix (list of lists)
        :param return: The hidden cell states
        """
        self.current_state = self.hidden_state
        states_series = []
        for current_input in self.inputs_series:
            current_input = tf.reshape(current_input, [self.settings.batch_size, 1])
            input_and_state_concatenated = tf.concat([current_input, self.current_state], 1)  # Increasing number of columns

            next_state = tf.tanh(tf.matmul(input_and_state_concatenated, self.W) + self.b)  # Broadcasted addition
            states_series.append(next_state)
            self.current_state = next_state
        return states_series
    # End of __forward_pass__()

    def __calculate_loss__(self):
        """
        Calculates the loss after a training epoch epoch.
        """
        states_series = self.__forward_pass__()
        logits_series = [tf.matmul(state, self.W2) + self.b2 for state in states_series] #Broadcasted addition
        self.predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

        losses = []
        for logits, labels in zip(logits_series, self.outputs_series):
            labels = tf.to_int32(labels, "CastLabelsToInt")
            losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        self.total_loss = tf.reduce_mean(losses)

        self.train_step = tf.train.AdagradOptimizer(0.3).minimize(self.total_loss)
        # return (total_loss, train_step)
    # End of __calculate_loss__()

    def plot(self, loss_list, prediction_series, batch_x, batch_y):
        """
        Plots a graph of the training output.
        """
        plt.subplot(2, 3, 1)
        plt.cla()
        plt.plot(loss_list)

        for batch_series_idx in range(5):
            one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
            single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

            plt.subplot(2, 3, batch_series_idx + 2)
            plt.cla()
            plt.axis([0, self.settings.truncate, 0, 2])
            left_offset = range(self.settings.truncate)
            plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
            plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
            plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

        plt.draw()
        plt.pause(0.0001)
    # End of plot()

    def __train_minibatch__(self, batch_num, epoch_num, sess):
        """
        Trains one minibatch.

        TODO: Refactor and improve comment.
        """
        self.logger.debug("Training minibatch : ", batch_num, " | ", "epoch : ", epoch_num + 1)
        start_index = batch_num * self.settings.truncate
        end_index = start_index + self.settings.truncate

        batch_x = self.x_train_batches[:, start_index:end_index]
        batch_y = self.y_train_batches[:, start_index:end_index]
        _total_loss, _train_step, _current_state, _predictions_series = sess.run(
            [self.total_loss, self.train_step, self.current_state, self.predictions_series],
            feed_dict={
                self.batch_x_placeholder:batch_x, 
                self.batch_y_placeholder:batch_y, 
                self.hidden_state:self._current_state
            })
        return _total_loss, _train_step, _current_state, _predictions_series
    # End of __train_minibatch__()

    def train(self):
        """
        Initial pseudocode for training the model.
        """
        self.logger.info("Started training the model.")
        self.__unstack_variables__()
        self.__create_batches__()
        self.__calculate_loss__() # Doesn't calculate actual loss, but rather creates symbolic functions that do so.
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            plt.ion()
            plt.figure()
            plt.show()
            loss_list = []

            for epoch_idx in range(self.settings.epochs):
                self._current_state = np.zeros((self.settings.batch_size, self.settings.hidden_size))
                self.logger.info("Starting epoch: %d" % (epoch_idx + 1))

                for batch_idx in range(self.num_batches):
                    _total_loss, _train_step, self._current_state, _predictions_series = self.__train_minibatch__(batch_idx, epoch_idx, sess)
                    loss_list.append(_total_loss)
                # End of batch training
                self.logger.info("Finished epoch: %d | loss: %f" % (epoch_idx + 1, _total_loss))
            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                plot(loss_list, _predictions_series, batchX, batchY)

        self.logger.info("Finished training the model. Final loss: %f" % _total_loss)
        plt.ioff()
        plt.show()
    # End of train()

    def generate_output(self):
        print("This feaure isn't implemented yet!")
    # End of generate_output()
