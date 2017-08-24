"""
Tensorflow implementation of a training method to train a given model.

Copyright (c) 2017 Frank Derry Wanye

Date: 24 August, 2017
"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

from . import constants
from .model import RNNModel

def __plot__(model, loss_list):
    """
    Plots a graph of epochs against losses.
    Currently, the loss_list stores data by minibatch, so it has to be traversed by
    skipping num_batches steps.

    :type loss_list: list()
    :param loss_list: the losses incurred during training.
    """
    epoch_loss_list = loss_list[::model.num_batches]
    plt.plot(range(1, len(epoch_loss_list) + 1), epoch_loss_list)
    plt.savefig(model.settings.log_dir + "plot.png")
    plt.show()
# End of plot()

def __train_minibatch__(model, batch_num, sess, current_state):
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
    start_index = batch_num * model.settings.truncate
    end_index = start_index + model.settings.truncate

    batch_x = model.x_train_batches[:, start_index:end_index]
    batch_y = model.y_train_batches[:, start_index:end_index]
    total_loss, train_step, current_state, predictions_series = sess.run(
        [model.total_loss_fun, model.train_step_fun, model.current_state, model.predictions_series],
        feed_dict={
            model.batch_x_placeholder:batch_x, 
            model.batch_y_placeholder:batch_y, 
            model.hidden_state_placeholder:current_state
        })
    return total_loss, current_state, predictions_series
# End of __train_minibatch__()

def __train_epoch__(model, epoch_num, sess, current_state, loss_list):
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
    model.logger.info("Starting epoch: %d" % (epoch_num))

    for batch_num in range(model.num_batches):
        # Debug log outside of function to reduce number of arguments.
        model.logger.debug("Training minibatch : ", batch_num, " | ", "epoch : ", epoch_num + 1)
        total_loss, current_state, predictions_series = __train_minibatch__(model, batch_num, sess, current_state)
        loss_list.append(total_loss)
    # End of batch training

    model.logger.info("Finished epoch: %d | loss: %f" % (epoch_num, total_loss))
    return total_loss, current_state, predictions_series
# End of __train_epoch__()

def train(model):
    """
    Trains the given model on the given dataset, and saves the losses incurred
    at the end of each epoch to a plot image.
    """
    model.logger.info("Started training the model.")
    model.__unstack_variables__()
    model.__create_functions__()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_list = []

        current_state = np.zeros((model.settings.batch_size, model.settings.hidden_size), dtype=float)
        for epoch_idx in range(1, model.settings.epochs + 1):
            total_loss, current_state, predictions_series = __train_epoch__(model, epoch_idx, sess, current_state, loss_list)
            # End of epoch training

    model.logger.info("Finished training the model. Final loss: %f" % total_loss)
    __plot__(model, loss_list)
# End of train()