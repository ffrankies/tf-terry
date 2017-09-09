"""
Tensorflow implementation of methods for generating RNN output.

Copyright (c) 2017 Frank Derry Wanye

Date: 5 September, 2017
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import math

from . import constants
from .model import RNNModel

def softmax(probabilities):
    """
    Compute softmax values for each sets of scores in the given array.
    It is needed because python's floats are dumb. 
    There are times when, say, 1.00000 does not equal 1
    Credit: https://stackoverflow.com/q/34968722/5760608

    :type probabilities: an array of floats.
    :param probabilities: an array of probabilities.

    :type return: an array of floats.
    :param return: an array of probabilities that sum up to 1.
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)
# End of softmax

def sentence_to_batch_array(model, sentence):
    """
    Convert sentence to batch array.

    :type sentence: list of int()
    :param sentence: the input sentence on which to generate output.

    :type return: np.array() 
    :param return: the correct slice of the input sentence, replicated across all batches.
    """
    batch_array = np.zeros((model.settings.truncate))
    start_position = math.floor((len(sentence)-1)/model.settings.truncate) * model.settings.truncate
    for index, word in enumerate(sentence[start_position:start_position+model.settings.truncate]):
        batch_array[index] = word
    # Make all rows the same (so it doesn't matter what batch is accessed afterwards)
    batch_array = np.tile(batch_array, (model.settings.batch_size, 1))
    return batch_array
# End of __sentence_to_batch_array__()

def predict(model, sentence, current_state):
    """
    Pass the input sentence through the RNN and retrieve the predictions.

    :type model: RNNModel()
    :param model: the model through which to pass the sentence.

    :type sentence: list of int()
    :param sentence: the sentence to pass throught the RNN.

    :type current_state: np.array()
    :param current_state: the current hidden_state of the RNN.

    :type return: (np.array(), np.array())
    :param return: the predictions and the hidden state after the sentence is passed through.
    """
    input_batch = sentence_to_batch_array(model, sentence)

    predictions, final_hidden_state = model.session.run(
        [model.predictions_series, model.current_state], 
        feed_dict={
            model.batch_x_placeholder:input_batch, 
            model.hidden_state_placeholder:current_state   
        })

    position = (len(sentence)-1) % model.settings.truncate
    return predictions[position][0], final_hidden_state
# End of predict()

def sample_output_word(model, probabilities):
    """
    Returns the probable next word in sentence. Some randomization is included to make sure 
    that not all the sentences produced are the same.

    :type model: RNNModel()
    :param model: the model on which the probabilities were calculated.

    :type probabilities: np.array()
    :param probabilities: the probabilities for the next word in the sentence.

    :type return: int()
    :param return: the index of the next word in the sentence.
    """
    output_word = model.word_to_index[constants.UNKNOWN]
    while output_word == model.word_to_index[constants.UNKNOWN]:
        while sum(probabilities[:-1]) > 1.0 : 
            model.logger.error("Sum of word probabilities (%f) > 1.0" % sum(probabilities[:-1]))
            probabilities = softmax(probabilities)
        samples = np.random.multinomial(1, probabilities)
        output_word = np.argmax(samples)
    return output_word
# End of sample_output_word()

def generate_output(model, num_tokens=-1, num_outputs=10):
    """
    Generates output from the RNN.

    :type model: RNNModel()
    :param model: the model used to generate the output.

    :type num_tokens: int()
    :param num_tokens: the number of tokens to generate. If set to -1, the output is generated until
                      an END token is reached.

    :type num_outputs: int()
    :param num_outputs: the number of outputs to generate.

    :type return: list of Strings
    :param return: a list of generated outputs in String format.
    """
    model.logger.info("Generating output.")
    outputs = []
    for i in range(1, num_outputs+1):
        sentence = generate_single_output(model, num_tokens)
        sentence = " ".join([model.index_to_word[word] for word in sentence])
        outputs.append(sentence)
    for sentence in outputs:
        print("%s\n\n" % sentence)
    model.logger.info("Generated outputs: %s" % outputs)
    return outputs
# End of generate_output()

def generate_single_output(model, num_tokens=-1):
    """
    Generates a single output from the model.

    :type model: RNNModel()
    :param model: the model used to generate the output.

    :type num_tokens: int()
    :param num_tokens: the number of tokens to generate. If set to -1, the output is generated until
                      an END token is reached.

    :type return: np.array()
    :param return: the generated output.
    """
    sentence = np.array([model.word_to_index[constants.SENT_START]])
    current_state = np.zeros((model.settings.batch_size, model.settings.hidden_size))
    num_tokens = 0
    while num_tokens < 1: # Generate until sentence_stop
        output, new_current_state = predict(model, sentence, current_state)
        sentence = np.append(sentence, sample_output_word(model, output))
        if sentence[-1] == model.word_to_index[constants.SENT_END] : break
        num_tokens -= 1
        num_tokens += 1
    # End of while loop
    return sentence
# End of __generate_single_output__()