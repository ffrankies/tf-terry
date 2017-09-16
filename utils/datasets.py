"""
This project is licensed under the MIT license:
https://github.com/ffrankies/Terry/blob/master/LICENSE

This project uses the Theano library:
https://arxiv.org/pdf/1605.02688.pdf

This project is based on the following tutorial:
https://goo.gl/DPf37h

Copyright (c) 2017 Frank Derry Wanye

Date: 22 Jul, 2017
"""

###############################################################################
# Source: rnn tutorial from www.wildml.com
# This script closely follows the tutorial, repurposing it to work with python3.
# This part of the code creates a dataset from a given csv file. The csv file
# should contain only one column, start with a column heading, and contain
# text data in sentence format. The script will break the data down into
# sentences, paragraphs or stories, tokenize them, and then save the dataset in
# a file of the user's choice.
# The file will contain the following items, in the same order:
#   the vocabulary of the training set
#   the vector used to convert token indexes to words
#   the dictionary used to convert words to token indexes
#   the input for training, in tokenized format (as indexes)
#   the output for training, in tokenized format (as indexes)
#
# Author: Frank Wanye
# Date: 13 September, 2017
###############################################################################

# Specify documentation format
__docformat__ = 'restructedtext en'

try:
    import _pickle as cPickle
except Exception:
    import cPickle
import re #regex library for re.split()
import os
import io
import numpy as np
import operator
import csv
import itertools
import nltk
import logging
import logging.handlers
import argparse
import time
from . import constants
from . import setup

def preprocess_data(logger, data_array):
    """
    Pre-processes data in data_array so that it is more or less modular.

    :type logger: logging.Logger()
    :param logger: the logger to which to write log output.

    :type data_array: list()
    :param data_array: the list of Strings to be preprocessed

    :type return: list()
    :param return: the list of preprocessed Strings.
    """
    logger.info("Preprocessing data")
    num_skipped = 0
    preprocessed_data = []
    for item in data_array:
        if "[" in item or "]" in item:
            num_skipped += 1
            continue
        item = item.replace("\n", " %s " % constants.CARRIAGE_RETURN)
        item = item.replace("\'\'", "\"")
        item = item.replace("``", "\"")
        preprocessed_data.append(item)
    logger.info("Skipped %d items in data." % num_skipped)
    return preprocessed_data
# End of preprocess_data()

def read_csv(logger, settings):
    """
    Reads the given csv file and extracts data from it into the comments array.
    Empty data cells are not included in the output.

    :type logger: logging.Logger
    :param logger: the logger to be used for logging

    :type settings: argparse.Namespace
    :param settings: the parse command-lien options to the program.

    :type return: List of Strings
    :param return: A List of comments to be converted into sentences, etc.
    """
    path = setup.get_arg(settings, 'raw_data_path', checkNone=True)

    # Encoding breaks when using python2.7 for some reason.
    comments = []
    logger.info("Reading the csv data file at: %s" % path)
    with open(path, "r", encoding='utf-8') as datafile:
        reader = csv.reader(datafile, skipinitialspace=True)
        try:
            reader.__next__() # Skips over table heading in Python 3.2+
        except Exception:
            reader.next() # For older versions of Python
        num_seen = 0
        for item in reader:
            if len(item) > 0 and len(item[0]) > 0:
                comments.append(item[0])
                num_seen += 1
                if (not settings.num_comments is None) and num_seen >= settings.num_comments:
                    break
        logger.info("%d examples kept for creating training dataset." % num_seen)
    # End with
    return comments
# End of read_csv()

def tokenize_sentences(logger, settings):
    """
    Uses the nltk library to break comments down into sentences, and then
    tokenizes the words in the sentences. Also appends the sentence start and
    end tokens to each sentence.

    :type logger: logging.Logger
    :param logger: the logger to be used for logging

    :type settings: argparse.Namespace
    :param settings: the parse command-lien options to the program.

    :type return: List of Strings
    :param return: A list of tokenized sentence strings
    """
    num_examples = setup.get_arg(settings, 'num_examples', asInt=True)
    comments = read_csv(logger, settings)

    logger.info("Breaking comments down into sentences.")
    sentences = itertools.chain(*[nltk.sent_tokenize(comment.lower()) for comment in comments])
    sentences = list(sentences)
    logger.info("%d sentences found in dataset." % len(sentences))

    sentences = preprocess_data(logger, sentences)

    if (not num_examples is None) and num_examples < len(sentences):
        logger.info("Reducing number of sentences to %d" % num_examples)
        sentences = sentences[:num_examples]

    logger.info("Adding sentence start and end tokens to sentences.")
    sentences = ["%s %s %s" % (constants.SENT_START, sentence, constants.SENT_END) for sentence in sentences]

    logger.info("Tokenizing words in sentences.")
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    sentences = list(sentences)
    return sentences
# End of tokenize_sentences()

def tokenize_paragraphs(logger, settings):
    """
    Uses the nltk library to break comments down into paragraphs, and then
    tokenizes the words in the paragraphs. Also appends the paragraph start and
    end tokens to each paragraph.

    :type logger: logging.Logger
    :param logger: the logger to be used for logging

    :type settings: argparse.Namespace
    :param settings: the parse command-lien options to the program.

    :type return: List of Strings
    :param return: A list of tokenized paragraph strings
    """
    num_examples = setup.get_arg(settings, 'num_examples', asInt=True)
    comments = read_csv(logger, settings)

    logger.info("Breaking comments down into paragraphs.")
    for comment in comments:
        paragraphs.extend(re.split('\n+', comment.lower()))
    logger.info("%d comments were broken down into %d paragraphs." % (len(comments), len(paragraphs)))

    paragraphs = preprocess_data(logger, paragraphs)

    if (not num_examples is None) and num_examples < len(paragraphs):
        logger.info("Reducing number of paragraphs to %d" % num_examples)
        paragraphs = paragraphs[:num_examples]

    logger.info("Adding paragraph start and end tokens to paragraphs.")
    paragraphs = ["%s %s %s" % (constants.PARA_START, paragraph, constants.PARA_END) for paragraph in paragraphs]

    logger.info("Tokenizing words in paragraphs.")
    paragraphs = [nltk.word_tokenize(paragraph) for paragraph in paragraphs]
    paragraphs = list(paragraphs)
    return paragraphs
# End of tokenize_paragraphs()

def tokenize_stories(logger, settings):
    """
    Uses the nltk library to word tokenize entire comments, assuming that
    each comment is its own story. Also appends the story start and end tokens
    to each story.

    :type logger: logging.Logger
    :param logger: the logger to be used for logging

    :type settings: argparse.Namespace
    :param settings: the parse command-lien options to the program.

    :type return: List of Strings
    :param return: A list of tokenized paragraph strings
    """
    num_examples = setup.get_arg(settings, 'num_examples', asInt=True)
    comments = read_csv(logger, settings)

    logger.info("Retrieving stories from data.")
    stories = [comment.lower() for comment in comments]
    logger.info("Found %d stories in the dataset." % len(stories))

    stories = preprocess_data(logger, stories)

    if (not num_examples is None) and num_examples < len(stories):
        logger.info("Reducing number of stories to %d" % num_examples)
        stories = stories[:num_examples]

    logger.info("Adding story start and end tokens to stories.")
    stories = ["%s %s %s" % (constants.STORY_START, story, constants.STORY_END) for story in stories]

    logger.info("Tokenizing words in stories.")
    stories = [nltk.word_tokenize(story) for story in stories]
    stories = list(stories)
    return stories
# End of tokenize_stories()

def create_dataset(logger, settings):
    """
    Creates a dataset using the tokenized data.

    :type logger: logging.Logger
    :param logger: the logger to be used for logging

    :type settings: argparse.Namespace
    :param settings: the parse command-lien options to the program.

    :type return: tuple
    :param return: (vocabulary as List, index_to_word as List, word_to_index as Dict, x_train as List, y_train as List)
    """
    mode = setup.get_arg(settings, 'mode', checkNone=True)
    vocab_size = setup.get_arg(settings, 'vocab_size', asInt=True, checkNone=True)
    if mode == 'sentences':
        data = tokenize_sentences(logger, settings)
    if mode == 'paragraphs':
        data = tokenize_paragraphs(logger, settings)
    if mode == 'stories':
        data = tokenize_stories(logger, settings)

    logger.info("Obtaining word frequency disribution.")
    word_freq = nltk.FreqDist(itertools.chain(*data))
    logger.info("Found %d unique words." % len(word_freq.items()))

    vocabulary = word_freq.most_common(vocab_size - 1)

    logger.info("Calculating percent of words captured...")
    total = 0
    for word in vocabulary:
        total += word_freq.freq(word[0])
    logger.info("Percent of total words captured: %f" % total * 100)

    index_to_word = [word[0] for word in vocabulary]
    index_to_word.append(constants.UNKNOWN)
    word_to_index = dict((word, index) for index, word in enumerate(index_to_word))

    logger.info("Replace all words not in vocabulary with unkown token.")
    for index, sentence in enumerate(data):
        data[index] = [word if word in word_to_index else constants.UNKNOWN for word in sentence]

    logger.info("Creating training data.")
    x_train = np.asarray([[word_to_index[word] for word in item[:-1]] for item in data])
    y_train = np.asarray([[word_to_index[word] for word in item[1:]] for item in data])

    return (vocabulary, index_to_word, word_to_index, x_train, y_train)
# End of create_dataset()

def save_dataset(logger, settings):
    """
    Saves the created dataset to a specified file.

    :type logger: logging.Logger
    :param logger: the logger to be used for logging

    :type settings: argparse.Namespace
    :param settings: the parse command-lien options to the program.
    """
    path = setup.get_arg(settings, 'saved_dataset_path', checkNone=True)
    filename = setup.get_arg(settings, 'saved_dataset_name', checkNone = True)

    setup.create_dir(path)
    with open(path + "/" + filename, "wb") as dataset_file:
        cPickle.dump(create_dataset(logger, settings), dataset_file, protocol=2)
# End of save_dataset()

def load_dataset(logger, settings):
    """
    Loads a saved dataset..

    :type logger: logging.Logger
    :param logger: the logger to be used for logging

    :type settings: argparse.Namespace
    :param settings: the parse command-lien options to the program.

    :type return: A tuple
    :param return: (vocabulary, index_to_word, word_to_index, x_train, y_train)
    """
    dataset_path = setup.get_arg(settings, 'saved_dataset_path', checkNone=True)
    dataset_name = setup.get_arg(settings, 'saved_dataset_name', checkNone=True)
    path = dataset_path + '/' + dataset_name

    logger.info("Loading saved dataset.")
    with open(path, "rb") as dataset_file:
        data = cPickle.load(dataset_file)
        vocabulary = data[0]
        index_to_word = data[1]
        word_to_index = data[2]
        x_train = data[3]
        y_train = data[4]

        logger.info("Size of vocabulary is: %d" % len(vocabulary))
        logger.info("Some words from vocabulary: %s" % index_to_word[:100])
        logger.info("Number of examples: %d" % len(x_train))
        logger.info("Sample training data: %s\n%s" % (x_train[:10], y_train[:10]))
    # End with
    return data
# End of load_dataset()

def parse_arguments():
    """
    Parses command-line arguments and returns the array of arguments.

    :type return: list
    :param return: list of parsed command-line arguments
    """
    arg_parse = argparse.ArgumentParser()
    setup.__add_log_arguments__(arg_parse)
    setup.__add_dataset_arguments__(arg_parse)
    arg_parse.add_argument("-v", "--vocab_size", default=8000, type=int,
                           help="The size of the dataset vocabulary.")
    arg_parse.add_argument("-c", "--num_comments", type=int,
                           help="The number of comments to be read.")
    arg_parse.add_argument("-n", "--num_examples", type=int,
                           help="The number of sentence examples to be saved.")
    arg_parse.add_argument("-e", "--test", action="store_true",
                           help="Specify if this is just a test run.")
    arg_parse.add_argument("-m", "--mode", default='sentences',
                           choices=['sentences', 'paragraphs', 'stories'],
                           help="Selects what constitutes an example in the "
                                "dataset.")
    return arg_parse.parse_args()
# End of parse_arguments()

def run():
    settings = parse_arguments()
    logger = logging.getLogger("datasets")
    logger_dir = "dataset_log/"
    logger.setLevel(logging.INFO)
    setup.create_dir(logger_dir)
    # Logger will use up to 5 files for logging, 'rotating' the data between them as they get filled up.
    handler = logging.handlers.RotatingFileHandler(
        filename=logger_dir + constants.LOGGING,
        maxBytes=1024*512,
        backupCount=5
    )
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    save_dataset(logger, settings)
    if settings.test:
        load_dataset(logger, settings)

if __name__ == "__main__":
    run()