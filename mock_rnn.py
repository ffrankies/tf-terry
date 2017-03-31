"""
This project is licensed under the MIT license:
https://github.com/ffrankies/Terry/blob/master/LICENSE

This project uses the Theano library:
https://arxiv.org/pdf/1605.02688.pdf

This project is based on the following tutorial:
https://goo.gl/DPf37h

Copyright (c) 2017 Frank Derry Wanye

Date: 26 March, 2017
"""
import numpy as np
try:
    import _pickle as cPickle
except Exception:
    import cPickle
import theano
import rnn.Mode
import logging
import logging.handlers

class MockRnn(object):
    """
    A mock RNN class, that does not use theano. This only has functions for
    generating output, since training without theano would be too costly.
    """

    def __init__(self, model="models/mid_story.pkl", trunc=500):
        """
        Creates a new MockRnn object, using the data stored in the model
        file.

        :type model: string
        :param model:
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

        self.load_parameters(model)
    # End of __init__()

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

            self.weights_eh = params[3]
            self.weights_hh = params[4]
            self.weights_ho = params[5]

            self.vocabulary = params[6]
            if self.unknown_token not in self.vocabulary:
                self.log.info("Appending unknown token")
                self.vocabulary[-1] = self.unknown_token
            self.index_to_word = params[7]
            self.word_to_index = params[8]

            self.bias = params[9]
            self.out_bias = params[10]

            self.embed_size = params[11]
            self.weights_emb = params[12]
    # End of load_parameters()

    def hard_sigmoid(self, x):
        """
        An approximation of sigmoid.
        More approximate and faster than ultra_fast_sigmoid.
        Approx in 3 parts: 0, scaled linear, 1
        Removing the slope and shift does not make it faster.

        Based on: https://goo.gl/q3aXZ2
        """
        slope = 0.2
        shift = 0.3
        x = (x * slope) + shift
        if x < 0:
            x = 0
        elif x > 1:
            x = 1
        return x
    # End of hard_sigmoid

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

        update_gate = self.hard_sigmoid(
            np.dot(self.weights_eh[0], embedding) +
            np.dot(self.weights_hh[0], previous_state) +
            self.bias[0]
        )

        reset_gate = self.hard_sigmoid(
            np.dot(self.weights_eh[1], embedding) +
            np.dot(self.weights_hh[1], previous_state) +
            self.bias[1]
        )

        hypothesis = self.hard_sigmoid(
            np.dot(self.weights_eh[2], embedding) +
            np.dot(self.weights_hh[2], previous_state * reset_gate) +
            self.bias[2]
        )

        temp = np.ones_like(update_gate) - update_gate
        current_state = temp * hypothesis + update_gate * previous_state

        # Softmax returns matrix with one row, so need to extract row from
        # matrix
        current_output = T.nnet.softmax(
            self.weights_ho.dot(current_state) + self.out_bias
        )[0]

        return [current_output, current_state]
