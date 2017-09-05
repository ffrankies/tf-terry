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

shown = False
shown2 = 0

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    Credit: https://stackoverflow.com/q/34968722/5760608
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)
# End of softmax

def __sentence_to_batch_array__(model, sentence):
    """
    Convert sentence to batch array.
    """
    batch_array = np.zeros((model.settings.truncate))
    start_position = math.floor((len(sentence)-1)/model.settings.truncate)*model.settings.truncate
    for index, word in enumerate(sentence[start_position:start_position+model.settings.truncate]):
        batch_array[index] = word
    # Make all rows the same (so it doesn't matter what batch is accessed afterwards)
    batch_array = np.tile(batch_array, (model.settings.batch_size, 1))
    return batch_array
# End of __sentence_to_batch_array__()

def run_step(model, sentence, sess, current_state):
    global shown

    input_batch = __sentence_to_batch_array__(model, sentence)

    predictions, final_hidden_state = sess.run(
        [model.predictions_series, model.current_state], 
        feed_dict={
            model.batch_x_placeholder:input_batch, 
            model.hidden_state_placeholder:current_state   
        })
    position = (len(sentence)-1) % model.settings.truncate
    return predictions[position][0], final_hidden_state
# End of run_step()

# def handle_bad_softmax(model, probabilities):
#     """
#     Handles bad softmax.
#     """
#     if sum(probabilities[:-1] > 1.0):
#         model.logger.error("Sum of word probabilities (%f) > 1.0" % sum(probabilities[:-1]))

def sample_output_word(model, probabilities):
    """
    Returns the probable next word in sentence.
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

def generate_output(model, session, initialize_variables, num_words_to_generate=-1, num_sentences_to_generate=100):
    """
    Generates output sentences.
    """
    model.logger.info("Generating output.")
    sentences = []
    if initialize_variables == True : session.run(tf.global_variables_initializer()) 
    for i in range(1, num_sentences_to_generate+1):
        sentence = __generate_single_output__(model, session, initialize_variables, )
        sentence = " ".join([model.index_to_word[word] for word in sentence])
        sentences.append(sentence)
    for sentence in sentences:
        print("%s\n\n" % sentence)
    model.logger.info("Generated sentences: %s" % sentences)
# End of generate_output()

def __generate_single_output__(model, session, num_words_to_generate=-1):
    """
    Generates a single output sentence/paragraph/word.
    """
    sentence = np.array([model.word_to_index[constants.SENT_START]])
    current_state = np.zeros((model.settings.batch_size, model.settings.hidden_size))
    num_words = 0
    while num_words_to_generate < 1: # Generate until sentence_stop
        output, new_current_state = run_step(model, sentence, session, current_state)
        sentence = np.append(sentence, sample_output_word(model, output))
        if sentence[-1] == model.word_to_index[constants.SENT_END] : break
        num_words_to_generate -= 1
        num_words += 1
    # End of while loop
    return sentence
# End of __generate_single_output__()

def save_output(model):
    """
    Saves sentence output to a file.
    """
    print("This feature isn't implemented yet!")
# End of save_output()