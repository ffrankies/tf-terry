"""
This project is licensed under the MIT license:
https://github.com/ffrankies/Terry/blob/master/LICENSE

This project uses the Theano library:
https://arxiv.org/pdf/1605.02688.pdf

This project is based on the following tutorial:
https://goo.gl/DPf37h

Copyright (c) 2017 Frank Derry Wanye

Date: 2 April, 2017
"""

###############################################################################
# Source: RNN tutorial from www.wildml.com
#
# The following is a python3 implementation of a Recurrent Neural Network with
# GRU units to predict the next word in a sentence. The nature of this
# network is that it can be used to generate sentences, words or paragraphs
# that will resemble the format of the training set. This particular version
# was meant to be used with reddit comment datasets, although it should in
# theory work fine with other datasets, as well.
# The various datasets that this particular RNN will be used for will be
# kernels of the main dataset of reddit comments from May 2015 fourd on kaggle
# Dataset source: https://www.kaggle.com/reddit/reddit-comments-may-2015
#
# Author: Frank Derry Wanye
# Date: 2 April 2017
###############################################################################

# Specify documentation format
__docformat__ = 'restructedtext en'

try:
    import _pickle as cPickle
except Exception:
    import cPickle
import theano
import theano.tensor as T
import numpy as np
import os
import sys
import time
import timeit
import logging
import logging.handlers
import argparse
import matplotlib
import re
matplotlib.use('Agg') # For saving of plot
from matplotlib import pyplot as plt # For plotting
from enum import Enum

class Mode(Enum):
    """
    An enum class for describing the mode of the RNN. The mode corresponds to
    what kind of output the RNN is meant to produce. Modes are:
    1 - SENTENCES
    2 - PARAGRAPHS
    3 - STORIES
    """
    SENTENCES = 1
    PARAGRAPHS = 2
    STORIES = 3
# End of Mode(Enum)

class GruRNN(object):
    """
    A Recurrent Neural Network that GRU units (neurons) in the hidden layer for
    combating vanishing and exploding gradients. The GRU is similar to the
    LSTM, but involves less computations, and is therefore more efficient. The
    network essentially looks like this, with the hidden units being GRU units:
        output      output      output      ...     output
          |           |           |                   |
    ----hidden------hidden------hidden------...-----hidden----
          |           |           |                   |
      embedding   embedding   embedding     ...   embedding
          |           |           |                   |
        input       input       input       ...     input
    The weights in this network are shared between the horizontal layers.
    The input and output for each horizontal layer are in the form of a vector
    (list, in the python implementation) of size len(vocabulary). For the
    input, the vector is a one-hot vector - it is comprised of all zeros,
    except for the index that corresponds to the input word, where the value is
    1. The output vector will contain the probabilities for each possible word
    being the next word. The word chosen as the actual output will the word
    corresponding to the index with the highest probability.
    """

    def __init__(self, hid_size=100, trunc=4, model=None,
                 dataset="reladred.pkl", emb_size=100):
        """
        Initializes the Vanilla RNN with the provided vocabulary_size,
        hidden layer size, and bptt_truncate. Also initializes the functions
        used in this RNN.

        :type hidden_size: int
        :param hidden_size: the number of hidden layer neurons in the network.
                            Default: 100

        :type trunc: int
        :param trunc: how far back back-propagation-through-time goes.
                      This is a crude method of reducing the effect
                      of vanishing/exploding gradients, as well as
                      reducing training time, since the network won't
                      have to go through every single horizontal layer
                      during training. NOTE: this causes the accuracy
                      of the network to decrease. Default: 4

        :type model: string
        :param model: the name of the saved model that contains all the RNN
                      info.

        :type dataset: string
        :param dataset: the pickled file containing the training data.
        """

        self.log = logging.getLogger("TEST.GRU")
        self.log.setLevel(logging.INFO)

        self.unknown_token = "UNKNOWN_TOKEN"
        self.sentence_start_token = "SENTENCE_START"
        self.sentence_end_token = "SENTENCE_END"
        self.paragraph_start = "PARAGRAPH_START"
        self.paragraph_end = "PARAGRAPH_END"
        self.story_start = "STORY_START"
        self.story_end = "STORY_END"

        self.load_data(dataset)

        if model is None:
            self.log.info("Initializing RNN parameters and functions...")
            self.hidden_size = hid_size
            self.bptt_truncate = trunc
            self.embed_size = emb_size

            # Instantiate the network weights
            # I feel like the first and third are switched for some reason...
            # but it's pretty consistent in the example code. Perhaps it's
            # backwards for a purpose
            # The weights going from input layer to the embedding layer
            # (E, in tutorial)
            weights_emb = np.random.uniform(-np.sqrt(1./self.vocabulary_size),
                                             np.sqrt(1./self.vocabulary_size),
                                             (emb_size, self.vocabulary_size))

            # The weights going from embedding layer to hidden layer
            # (U, in tutorial)
            weights_eh = np.random.uniform(-np.sqrt(1./self.vocabulary_size),
                                            np.sqrt(1./self.vocabulary_size),
                                            (3, hid_size, emb_size))
            # The weights going from hidden layer to hidden layer
            # (W, in tutorial)
            weights_hh = np.random.uniform(-np.sqrt(1./self.vocabulary_size),
                                            np.sqrt(1./self.vocabulary_size),
                                            (3, hid_size, hid_size))
            # The weights going from hidden layer to output layer
            # (V, in tutorial)
            weights_ho = np.random.uniform(-np.sqrt(1./self.vocabulary_size),
                                            np.sqrt(1./self.vocabulary_size),
                                            (self.vocabulary_size, hid_size))
            # The bias for the hidden units
            bias = np.zeros((3, hid_size))
            # The bias for the output units
            out_bias = np.zeros(self.vocabulary_size)

            self.weights_emb = theano.shared(
                name='weights_emb',
                value=weights_emb.astype(theano.config.floatX))

            self.weights_eh = theano.shared(
                name='weights_eh',
                value=weights_eh.astype(theano.config.floatX))

            self.weights_hh = theano.shared(
                name='weights_hh',
                value=weights_hh.astype(theano.config.floatX))

            self.weights_ho = theano.shared(
                name='weights_ho',
                value=weights_ho.astype(theano.config.floatX))

            self.bias = theano.shared(
                name='bias',
                value=bias.astype(theano.config.floatX))

            self.out_bias = theano.shared(
                name='out_bias',
                value=out_bias.astype(theano.config.floatX))
        else:
            self.load_parameters(model)
        # End of if statement

        self.log.info("Network parameters: \n"
                      "Vocabulary size: %d\n"
                      "Embedding layer size: %d\n"
                      "Hidden layer size: %d\n"
                      "Bptt truncate: %d\n" %
                      (self.vocabulary_size,
                       self.embed_size,
                       self.hidden_size,
                       self.bptt_truncate,))

        # Symbolic representation of one input sentence
        input = T.ivector('sentence')

        # Symbolic representation of the one output sentence
        output = T.ivector('sentence')

        def forward_propagate(word, previous_state):
            """
            Vertically propagates one of the words.

            :type word: int
            :param word: the index of the current input word

            :type previous_state: T.dvector()
            :param word: the output of the hidden layer from the previous
                         horizontal layer
            """
            # Word Embedding Layer
            embedding = self.weights_emb[:, word]

            # GRU layer
            update_gate = T.nnet.hard_sigmoid(
                self.weights_eh[0].dot(embedding) +
                self.weights_hh[0].dot(previous_state) +
                self.bias[0]
            )

            reset_gate = T.nnet.hard_sigmoid(
                self.weights_eh[1].dot(embedding) +
                self.weights_hh[1].dot(previous_state) +
                self.bias[1]
            )

            hypothesis = T.tanh(
                self.weights_eh[2].dot(embedding) +
                self.weights_hh[2].dot(previous_state * reset_gate) +
                self.bias[2]
            )

            temp = T.ones_like(update_gate) - update_gate
            current_state = temp * hypothesis + update_gate * previous_state

            # Softmax returns matrix with one row, so need to extract row from
            # matrix
            current_output = T.nnet.softmax(
                self.weights_ho.dot(current_state) + self.out_bias
            )[0]

            return [current_output, current_state]

        #######################################################################
        # Symbolically represents going through each input sentence word and
        # then calculating the state of the hidden layer and output word for
        # each word. The forward_propagate function is the one used to
        # generate the output word and hidden layer state.
        #######################################################################
        self.theano = {}

        [out, state], updates = theano.scan(
            forward_propagate,
            sequences=input,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          dict(initial=T.zeros(self.hidden_size))]
        )

        # Predicts the output words for each word in the sentence
        prediction = T.argmax(out, axis=1)

        # Calculates the output error between the predicted output and the
        # actual output
        # Categorical crossentropy formula according to deeplearning.net:
        # -Sum(p(x)log(q(x))), where p(x) is the true distribution and q(x) is
        # the calculated distribution
        out_error = T.sum(T.nnet.categorical_crossentropy(
            out + T.mean(out)/1000, output))

        # Symbolically represents gradient calculations for gradient descent
        d_weights_emb = T.grad(out_error, self.weights_emb)
        d_weights_eh = T.grad(out_error, self.weights_eh)
        d_weights_hh = T.grad(out_error, self.weights_hh)
        d_weights_ho = T.grad(out_error, self.weights_ho)
        d_bias = T.grad(out_error, self.bias)
        d_out_bias = T.grad(out_error, self.out_bias)

        # Symbolic theano functions
        self.forward_propagate = theano.function([input], out)
        self.predict = theano.function([input], prediction)
        self.calculate_error = theano.function([input, output], out_error)
        self.bptt = theano.function([input, output],
            [d_weights_emb, d_weights_eh, d_weights_hh, d_weights_ho, d_bias])

        # Stochastic Gradient Descent step
        learning_rate = T.scalar('learning_rate')

        emb_update = self.weights_emb - learning_rate * d_weights_emb
        eh_update = self.weights_eh - learning_rate * d_weights_eh
        hh_update = self.weights_hh - learning_rate * d_weights_hh
        ho_update = self.weights_ho - learning_rate * d_weights_ho
        bias_update = self.bias - learning_rate * d_bias
        out_bias_update = self.out_bias - learning_rate * d_out_bias

        self.sgd_step = theano.function(
            [input, output, learning_rate], [],
            updates=[(self.weights_emb, emb_update),
                     (self.weights_eh, eh_update),
                     (self.weights_hh, hh_update),
                     (self.weights_ho, ho_update),
                     (self.bias, bias_update),
                     (self.out_bias, out_bias_update)]
        )
    # End of __init__()

    def calculate_total_loss(self, train_x, train_y):
        """
        Calculates the sum of the losses for a given epoch (sums up the losses
        for each sentence in train_x).

        :type train_x: T.imatrix()
        :param train_x: the training examples (list of tokenized and indexed
                        sentences, starting from SENTENCE_START and not
                        including SENTENCE_END)

        :type train_y: T.imatrix()
        :param train_y: the training solutions (list of tokenized and indexed
                        sentences, not including SENTNECE_START and going to
                        SENTENCE_END)
        """
        return np.sum([self.calculate_error(x, y)
                       for x, y in zip(train_x, train_y)])
    # End of calculate_total_loss()

    def calculate_loss(self, train_x, train_y):
        """
        Calculates the average loss for a given epoch (the average of the
        output of calculate_total_loss())

        :type train_x: T.imatrix()
        :param train_x: the training examples (list of tokenized and indexed
                        sentences, starting from SENTENCE_START and not
                        including SENTENCE_END)

        :type train_y: T.imatrix()
        :param train_y: the training solutions (list of tokenized and indexed
                        sentences, not including SENTNECE_START and going to
                        SENTENCE_END)
        """
        self.log.info("Calculating average categorical crossentropy loss...")

        num_words = np.sum([len(y) for y in train_y])
        return self.calculate_total_loss(train_x, train_y)/float(num_words)
    # End of calculate_loss()

    def load_data(self, filePath="reladred.pkl"):
        """
        Loads previously saved data from a pickled file.
        The number of vocabulary words must match self.vocabulary_size - 1 or
        else the dataset will not work.

        :type filePath: string
        :param filePath: the path to the file containing the dataset.
        """
        self.log.info("Loading the dataset from %s" % filePath)

        with open(filePath, "rb") as data_file:
            vocab, i_to_w, w_to_i, x_train, y_train = cPickle.load(data_file)

            self.vocabulary = vocab
            self.index_to_word = i_to_w
            self.word_to_index = w_to_i
            self.x_train = x_train
            self.y_train = y_train

            if self.unknown_token not in self.vocabulary:
                self.vocabulary.append(self.unknown_token)

            self.vocabulary_size = len(self.vocabulary)
            self.log.info("Dataset contains %d words" % self.vocabulary_size)
    # End of calculate_loss()

    def load_parameters(self, model=None):
        """
        Loads the network parameters from a previously pickled file using the
        cPickle library.

        :type path: string
        :param path: the path to the pickled file containing the network
                     parameters.
        """
        self.log.info("Loading model parameters from saved model...")

        if model is None or model == "":
            self.log.error("No model provided.")
            sys.exit(1)

        with open(model, "rb") as modelFile:
            params = cPickle.load(modelFile)

            self.vocabulary_size = params[0]
            self.hidden_size = params[1]
            self.bptt_truncate = params[2]

            weights_eh = params[3]
            weights_hh = params[4]
            weights_ho = params[5]

            self.vocabulary = params[6]
            if self.unknown_token not in self.vocabulary:
                self.log.info("Appending unknown token")
                self.vocabulary[-1] = self.unknown_token
            self.index_to_word = params[7]
            self.word_to_index = params[8]

            bias = params[9]
            out_bias = params[10]

            self.embed_size = params[11]
            weights_emb = params[12]

            self.weights_emb = theano.shared(
                name='weights_emb',
                value=weights_emb.astype(theano.config.floatX))

            self.weights_eh = theano.shared(
                name='weights_eh',
                value=weights_eh.astype(theano.config.floatX))

            self.weights_hh = theano.shared(
                name='weights_hh',
                value=weights_hh.astype(theano.config.floatX))

            self.weights_ho = theano.shared(
                name='weights_ho',
                value=weights_ho.astype(theano.config.floatX))

            self.bias = theano.shared(
                name='bias',
                value=bias.astype(theano.config.floatX))

            self.out_bias = theano.shared(
                name='out_bias',
                value=out_bias.astype(theano.config.floatX))
    # End of load_parameters()

    def save_parameters(self, path=None, epoch=0):
        """
        Saves the network parameters using the cPickle library.

        :type path: string
        :param path: the path to which the parameters will be saved.
        """
        params = (
            self.vocabulary_size,
            self.hidden_size,
            self.bptt_truncate,
            self.weights_eh.get_value(),
            self.weights_hh.get_value(),
            self.weights_ho.get_value(),
            self.vocabulary,
            self.index_to_word,
            self.word_to_index,
            self.bias.get_value(),
            self.out_bias.get_value(),
            self.embed_size,
            self.weights_emb.get_value()
        )

        # Save embeddings separately, so they can be accessed later
        embed_params = (self.embed_size, self.weights_emb.get_value())

        if path is None:
            modelPath = "model.pkl"
            with open(modelPath, "wb") as file:
                cPickle.dump(params, file, protocol=2)

            embedPath = "embed.pkl"
            with open(modelPath, "wb") as file:
                cPickle.dump(embed_params, file, protocol=2)
        else:
            with open(path + ".pkl", "wb") as file:
                cPickle.dump(params, file, protocol=2)

            with open(path + "embed.pkl", "wb") as file:
                cPickle.dump(embed_params, file, protocol=2)
    # End of save_parameters()

    def save_graph(self, path=None, losses=None):
        """
        Saves a graph of epochs (x-axis) vs loss (y-axis) using the list of
        losses 'losses' in the file in path 'path'.

        :type path: String
        :param path: the path to the file where the graph will be saved.

        :type losses: list
        :param losses: a list of losses, one per epoch.
        """
        if losses is None:
            self.log.error("No losses to save.")
            return

        # Plot a graph of loss against epochs, save graph
        self.log.info("Plotting a graph of loss vs iteration")
        plot_iterations = []
        plot_losses = []
        for i in range(len(losses)):
            plot_iterations.append(i)
            plot_losses.append(losses[i][1])
        plt.plot(plot_iterations, plot_losses)
        if path is None:
            modelPath = "loss_plot.png"
            plt.savefig(modelPath)
        else:
            modelPath = path + "loss_plot.png"
            plt.savefig(modelPath)
    # End of save_graph()

    def train_rnn(self, learning_rate=0.005, epochs=1, patience=10000,
                  path=None, max=None, testing=False, anneal=0.000001):
        """
        Trains the RNN using stochastic gradient descent. Saves model
        parameters after every epoch.

        :type learning_rate: float
        :param learning_rate: multiplier for how much the weights of the
                              network get adjusted during gradient descent

        :type epochs: int
        :param epochs: the number of epochs (iterations over entire dataset)
                       for which the network will be trained.

        :type patience: int
        :param patience: the number of examples after which the loss should be
                         measured.

        :type path: string
        :param path: the path to the file in which the models should be stored.
                     The epoch number and .pkl extension will be added to the
                     path automatically.

        :type max: int
        :param max: the maximum number of examples it from the training set
                    used in the training.

        :type testing: bool
        :param testing: if this is set to true, the models will not be saved
                        off.

        :type anneal: float
        :param anneal: the minimum value to which the learning rate can be
                       annealed.
        """
        if self.x_train is None or self.y_train is None:
            self.log.info("Need to load data before training the rnn")

            return

        # Keep track of losses so that they can be plotted
        start_time = timeit.default_timer()

        losses = []
        examples_seen = 0

        # Evaluate loss before training
        self.log.info("Evaluating loss before training.")

        if max is None:
            loss = self.calculate_loss(self.x_train, self.y_train)
        else:
            loss = self.calculate_loss(self.x_train[:max],
                                       self.y_train[:max])

        losses.append((examples_seen, loss))

        self.log.info("RNN incurred a loss of %f before training" % loss)

        for e in range(epochs):
            epoch = e + 1
            self.log.info("Training the model: epoch %d" % epoch)


            # Train separately for each training example (no need for
            # minibatches)
            if max is None:
                for example in range(len(self.y_train)):
                    self.sgd_step(self.x_train[example], self.y_train[example],
                                  learning_rate)
                    examples_seen += 1
                    if examples_seen % patience == 0:
                        self.log.info("Evaluated %d examples" % examples_seen)

            else:
                for example in range(len(self.y_train[:max])):
                    self.sgd_step(self.x_train[example], self.y_train[example],
                                  learning_rate)
                    examples_seen += 1
                    if examples_seen % patience == 0:
                        self.log.info("Evaluated %d examples" % examples_seen)
            # End of training for epoch

            #Preventing zeros in weights
            #Retrieve numpy arrays of each network parameter
            # local_w_eh = self.weights_eh.get_value()
            # local_w_hh = self.weights_hh.get_value()
            # local_w_ho = self.weights_ho.get_value()
            # local_bias = self.bias.get_value()
            # local_out_bias = self.out_bias.get_value()
            #
            # out = self.forward_propagate(self.x_train[3])
            # print("%f - %f, %f" % (np.max(out), np.min(out), np.sum(out)))
            #
            # print("Max weight: %f, min weight: %f" %
            #     (np.max([
            #         np.max(local_w_eh),
            #         np.max(local_w_hh),
            #         np.max(local_w_ho),
            #         np.max(local_bias),
            #         np.max(local_out_bias)]),
            #      np.min([
            #         np.min(local_w_eh),
            #         np.min(local_w_hh),
            #         np.min(local_w_ho),
            #         np.min(local_bias),
            #         np.min(local_out_bias)])))
            #
            # print("Mean out: %f" % np.mean(out))
            #
            # if np.isnan(local_bias).any() or np.isnan(local_out_bias).any() or np.isnan(local_w_eh).any() or np.isnan(local_w_hh).any() or np.isnan(local_w_ho).any():
            #     print("Found a nan inside the weights.")
            #
            # if np.isinf(local_bias).any() or np.isinf(local_out_bias).any() or np.isinf(local_w_eh).any() or np.isinf(local_w_hh).any() or np.isinf(local_w_ho).any():
            #     print("Found an infinity inside the weights.")

            # Alternative - use theano.tensor.clip and switch to switch zeros
            # for small numbers, and clip numbers to a given max and min value

            # Evaluate loss after every epoch
            self.log.info("Evaluating loss: epoch %d" % epoch)

            if max is None:
                loss = self.calculate_loss(self.x_train, self.y_train)
            else:
                loss = self.calculate_loss(self.x_train[:max],
                                           self.y_train[:max])
            losses.append((examples_seen, loss))
            self.log.info("RNN incurred a loss of %f after %d epochs" %
                  (loss, epoch))

            # End of loss evaluation

            # Adjust learning rate if loss increases or stays almost constant
            if (len(losses) > 1 and losses[-1][1] >= 0.99 * losses[-2][1]):
                if learning_rate > anneal:
                    learning_rate = learning_rate * 0.5
                    self.log.info("Setting learning rate to %f" % learning_rate)

            # End training if incurred a loss of 0
            if losses[-1][1] == 0:
                end_time = timeit.default_timer()

                self.log.info(
                    ("Finished training the rnn for %d epochs, with a final" +
                     " loss of %f. Training took %.2f m") %
                    (epochs, losses[-1][1], (end_time - start_time) / 60)
                )

                self.save_graph(path=path, losses=losses)

                return (0, (end_time - start_time) / 60)

            # End training if loss no longer decreasing significantly
            if len(losses) > 10 and losses[-1][1] >= 0.99 * losses[-10][1]:
                end_time = timeit.default_timer()

                self.log.info(
                    ("Training ended because the loss is no longer "
                     "significantly decreasing. Training took %d epochs, with "
                     "a final loss of %f. Training took %.2f m") %
                    (epochs, losses[-1][1], (end_time - start_time) / 60)
                )

                self.save_graph(path=path, losses=losses)

                return (0, (end_time - start_time) / 60)

            # Saving model parameters
            if testing == False:
                self.save_parameters(path, epoch)
        # End of training

        end_time = timeit.default_timer()

        self.log.info(
            ("Finished training the rnn for %d epochs, with a final loss of %f."
             + " Training took %.2f m") %
            (epochs, losses[-1][1], (end_time - start_time) / 60)
        )

        self.save_graph(path=path, losses=losses)

        return (losses[-1][1], (end_time - start_time) / 60)
    # End of train_rnn()

    def generate_sentence(self):
        """
        Generates one sentence based on current model parameters. Model needs
        to be loaded or trained before this step in order to produce any
        results.

        :return type: list of strings
        :return param: a generated sentence, with each word being an item in
                       the array.
        """
        if self.word_to_index is None:
            self.log.info("Need to load a model or data before this step.")
            return []

        num_predictions = 0
        num_unknowns = 0

        # Start sentence with the start token
        sentence = [self.word_to_index[self.sentence_start_token]]
        # Predict next word until end token is received
        while not sentence[-1] == self.word_to_index[self.sentence_end_token]:
            num_predictions += 1
            next_word_probs = self.forward_propagate(sentence)
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
            # We don't want the unknown token to appear in the sentence
            while sampled_word == self.word_to_index[self.unknown_token]:
                num_unknowns += 1
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)
            sentence.append(sampled_word)
        sentence_str = [self.index_to_word[word] for word in sentence[1:-1]]
        return (sentence_str, (num_unknowns / num_predictions) * 100)
    # End of generate_sentence()

    def generate_paragraph(self):
        """
        Generates one paragraph based on current model parameters. Model needs
        to be loaded or trained before this step in order to produce any
        results.

        :return type: list of strings
        :return param: a generated paragraph, with each word being an item in
                       the array.
        """
        if self.word_to_index is None:
            self.log.info("Need to load a model or data before this step.")
            return []

        num_predictions = 0
        num_unknowns = 0

        # Start paragraph with the start token
        paragraph = [self.word_to_index[self.paragraph_start]]
        # Predict next word until end token is received
        while not paragraph[-1] == self.word_to_index[self.paragraph_end]:
            num_predictions += 1
            next_word_probs = self.forward_propagate(paragraph)
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
            # We don't want the unknown token to appear in the paragraph
            while sampled_word == self.word_to_index[self.unknown_token]:
                num_unknowns += 1
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)
            paragraph.append(sampled_word)
        paragraph_str = [self.index_to_word[word] for word in paragraph[1:-1]]
        return (paragraph_str, (num_unknowns / num_predictions) * 100)
    # End of generate_paragraph()

    def generate_story(self, min_length=500):
        """
        Generates one story based on current model parameters. Model needs
        to be loaded or trained before this step in order to produce any
        results.

        :type min_length: int
        :param min_length: the minimum length of the generated story

        :return type: list of strings
        :return param: a generated story, with each word being an item in
                       the array.
        """
        if self.word_to_index is None:
            self.log.info("Need to load a model or data before this step.")
            return []

        num_predictions = 0
        num_unknowns = 0

        unknown = self.word_to_index[self.unknown_token]
        end = self.word_to_index[self.story_end]
        # Start story with the start token
        story = [self.word_to_index[self.story_start]]

        # Ensuring minimum length of story is met.
        while len(story)-1 < min_length:
            num_predictions += 1
            next_word_probs = self.forward_propagate(story)
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
            # We don't want the unknown token to appear in the story
            while sampled_word == unknown or sampled_word == end:
                num_unknowns += 1
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)
            # End of while loop
            story.append(sampled_word)

        # After minimum length requirement met, reach 'natural' conclusion.
        # Predict next word until end token is received
        while not story[-1] == self.word_to_index[self.story_end]:
            num_predictions += 1
            next_word_probs = self.forward_propagate(story)
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
            # We don't want the unknown token to appear in the story
            while sampled_word == unknown:
                num_unknowns += 1
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)
            # End of while loop
            story.append(sampled_word)

        story_str = [self.index_to_word[word] for word in story[1:-1]]
        return (story_str, (num_unknowns / num_predictions) * 100)
    # End of generate_story()

    def generate_output(self, mode=Mode.STORIES, num=25, path="out.txt",
                        minLength=0):
        """
        Saves 'num' of RNN output of type 'mode' to the file with path 'path'.

        :type mode: Mode Enum
        :param mode: The Mode of the RNN (the type of output it produces).

        :type num: int
        :param num: The number of outputs to save to file.

        :type path: String
        :param path: The path to the file in which the output will be saved.

        :type minLength: int
        :param minLength: The minimum length of the output to be saved.
        """
        attempts = 0
        successes = 0
        percent_unknowns = 0

        with open(path, "w") as outFile:

            while successes < num:
                if mode is Mode.SENTENCES:
                    output = self.generate_sentence()
                if mode is Mode.PARAGRAPHS:
                    output = self.generate_paragraph()
                if mode is Mode.STORIES:
                    output = self.generate_story()

                percent_unknowns += output[1]
                output = output[0]

                if len(output) > minLength:
                    outFile.write(self.pretty_print(output) + "\n\n\n")
                    successes += 1

                attempts += 1
            # End of while loop
        # End of file usage

        self.log.info("Generated %d outputs after %d attempts." %
                     (successes, attempts))

        percent_unknowns = percent_unknowns / attempts
        self.log.info("%f percent of the words generated were unknown tokens." %
                     percent_unknowns)
    # End of generate_output()

    def pretty_print(self, output):
        """
        Converts RNN output into a string.
        Got some help from: http://stackoverflow.com/a/29044417/5760608

        :type output: list of strings
        :param output: generated output.
        """
        if "SPACE_TOKEN" in output:
            string = "".join(output)
            string = string.replace("SPACE_TOKEN", " ")
        else:
            string = " ".join(output)
        string = string.replace("CARRIAGE_RETURN", "\n")
        string = string.replace("\'\'", "\"")
        string = string.replace("``", "\"")
        string = string.replace(" .", ".")
        string = string.replace(" ,", ",")
        string = string.replace(" !", "!")
        string = string.replace(" ?", "?")
        string = re.split('([.!?] *)', string)
        string = [s.capitalize() for s in string]
        string = "".join(string)
        string = string.replace(" i ", " I ")
        string = string.replace(" \'re", "\'re")
        string = string.replace(" \'d", "\'d")
        string = string.replace(" \'m", "\'m")
        string = string.replace(" n\'t", "n\'t")
        string = string.replace(" \'s", "\'s")
        return string
# End of GruRNN(object)

def createDir(dirPath):
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
# End of createDir()

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

if __name__ == "__main__":
    args = parse_arguments()

    argsdir = args.dir + "/" + time.strftime("%d%m%y%H") + "/";
    sentenceDir = argsdir
    modelDir = argsdir
    logDir = argsdir
    logFile = args.filename

    createDir(sentenceDir)
    createDir(modelDir)
    createDir(logDir)

    testlog = logging.getLogger("TEST")
    testlog.setLevel(logging.INFO)

    handler = logging.handlers.RotatingFileHandler(
        filename=logDir+logFile,
        maxBytes=1024*512,
        backupCount=5
    )

    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(message)s"
    )

    handler.setFormatter(formatter)

    testlog.addHandler(handler)
    testlog.info("Running a new GRU-RNN with logging")

    RNN = GruRNN(
        model=args.model,
        trunc=args.truncate,
        hid_size=args.hidden_size,
        dataset=args.dataset,
        emb_size=args.embed_size
    )

    RNN.train_rnn(
        epochs=args.epochs,
        patience=args.patience,
        path=modelDir+"reladred",
        max=args.max,
        testing=args.test,
        learning_rate=args.learn_rate,
        anneal=args.anneal
    )

    if args.test:
        testlog.info("Finished running test.")
        sys.exit(0)

    testlog.info("Generating output")

    if RNN.sentence_start_token in RNN.word_to_index:
        mode=Mode.SENTENCES
    elif RNN.paragraph_start in RNN.word_to_index:
        mode=Mode.PARAGRAPHS
    elif RNN.story_start in RNN.word_to_index:
        mode=Mode.STORIES
    RNN.generate_output(path=sentenceDir + "out.txt", mode=mode)
