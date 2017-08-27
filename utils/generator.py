"""
Tensorflow implementation of methods for generating RNN output.

Copyright (c) 2017 Frank Derry Wanye

Date: 25 August, 2017
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging

from . import constants
from .model import RNNModel

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
    #TODO: Move sentence to END of batch array. That way, we can always pick the element at the end of the probabilities_series.
    batch_array = np.zeros((model.settings.batch_size, model.settings.truncate))
    for index, word in enumerate(sentence[-1:-11:-1]):
        batch_array[0][-index] = word
    return batch_array
# End of __sentence_to_batch_array__()

def run_step(model, sentence, sess, word_index):
    input_batch = __sentence_to_batch_array__(model, sentence)

    output, current_state = sess.run(
        [model.predictions_series, model.current_state], 
        feed_dict={
            model.batch_x_placeholder:input_batch, 
            model.hidden_state_placeholder:model.latest_state   
        })

    print("Shape of predictions_series (output). Height: ", len(output), " Width: ", len(output[0]), " Breadth?: ", len(output[0][0]))
    print("Shape of current_state: ", current_state.shape)
    print("Output[-1]", output[-1][0])
    return output[-1][0]
# End of run_step()

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

def __generate_single_output__(model, session, num_words_to_generate):
    """
    Generates a single output sentence/paragraph/word.
    """
    sentence = np.array([model.word_to_index[constants.SENT_START]])
    # print("sentence: ", sentence)
    word_index = 0
    while num_words_to_generate < 1: # Generate until sentence_stop
        output = run_step(model, sentence, session, word_index)
        sentence = np.append(sentence, sample_output_word(model, output))
        # print("Sentence: ", " ".join([model.index_to_word[word] for word in sentence]))
        if sentence[-1] == model.word_to_index[constants.SENT_END] : break
        if word_index < model.settings.truncate - 1 : word_index += 1
    # print("Final sentence: ", " ".join([model.index_to_word[word] for word in sentence]))
    return sentence
# End of __generate_single_output__()

def save_output(model):
    """
    Saves sentence output to a file.
    """
    print("This feature isn't implemented yet!")
# End of save_output()