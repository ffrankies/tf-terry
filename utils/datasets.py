"""
This project is licensed under the MIT license:
https://github.com/ffrankies/Terry/blob/master/LICENSE

This project uses the Theano library:
https://arxiv.org/pdf/1605.02688.pdf

This project is based on the following tutorial:
https://goo.gl/DPf37h

Copyright (c) 2017 Frank Derry Wanye

Date: 19 Jul, 2017
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
# Date: 19 July, 2017
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
from . import setup
from . import constants

###############################################################################
# Setting up global variables
###############################################################################
log = None # Logging
timestr = time.strftime("%d%m%y%H") # For when current time is needed
comments = [] # Holds the comments
sentences = [] # Holds the sentences from the comments
paragraphs = [] # Holds the paragraphs from the comments
stories = [] # Holds the stories
vocab_size = 0 # Number of words RNN wil remember
# Special tokens
unknown = "UNKNOWN_TOKEN"
sentence_start = "SENTENCE_START"
sentence_end = "SENTENCE_END"
paragraph_start = "PARAGRAPH_START" # Use split on (\n or \n\n) for this?
paragraph_end = "PARAGRAPH_END"
story_start = "STORY_START"
story_end = "STORY_END"
space_token = "SPACE_TOKEN"
enter = "CARRIAGE_RETURN"
# Dataset parameters
vocabulary = []
word_to_index = []
index_to_word = []
x_train = []
y_train = []

def read_csv(path=None, max=None):
    """
    Reads the given csv file and extracts data from it into the comments array.
    Empty data cells are not included in the output.

    :type path: String
    :param path: the path to the csv data file
    """
    global log
    global comments

    if path is None:
        path = input("Enter path to the scv data file: ")

    # Encoding breaks when using python2.7 for some reason.
    log.info("Reading the csv data file at: %s" % path)
    with open(path, "r", encoding='utf-8') as datafile:
        reader = csv.reader(datafile, skipinitialspace=True)
        try:
            reader.__next__() # Skips over table heading in Python 3.2+
        except Exception:
            reader.next() # For older versions of Python
        num_seen = 0
        num_saved = 0
        for item in reader:
            if len(item) > 0 and len(item[0]) > 0:
                comments.append(item[0])
                num_saved += 1
                if (not max is None) and num_saved >= max:
                    num_seen += 1
                    log.info("Gone over %d examples, saved %d of them" %
                             (num_seen, num_saved))
                    break
            num_seen += 1
        log.info("Gone over %d examples, saved %d of them" %
                 (num_seen, num_saved))
# End of read_csv()

def tokenize_sentences(num_examples=None):
    """
    Uses the nltk library to break comments down into sentences, and then
    tokenizes the words in the sentences. Also appends the sentence start and
    end tokens to each sentence.
    """
    global sentences
    global comments
    global sentence_start
    global sentence_end
    global log
    global space_token

    log.info("Breaking comments down into sentences.")
    sentences = itertools.chain(
        *[nltk.sent_tokenize(comment.lower()) for comment in comments])
    sentences = list(sentences)
    log.info("%d sentences found in dataset." % len(sentences))

    log.info("Preprocessing sentences")
    sents = []
    for item in sentences:
        item = item.replace(" ", " %s " % space_token)
        item = item.replace("\'\'", "\"")
        item = item.replace("``", "\"")
        sents.append(item)
    sentences = sents

    for sentence in sentences:
        if "[" in sentence or "]" in sentence:
            sentences.remove(sentence)

    if (not num_examples is None) and num_examples < len(sentences):
        log.info("Reducing number of sentences to %d" % num_examples)
        sentences = sentences[:num_examples]

    log.info("Adding sentence start and end tokens to sentences.")
    sentences = ["%s %s %s" % (sentence_start, sentence, sentence_end)
                 for sentence in sentences]

    log.info("Tokenizing words in sentences.")
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    sentences = list(sentences)
# End of tokenize_sentences()

def create_sentence_dataset(vocab_size=8000):
    """
    Creates a dataset using the tokenized sentences.

    :type vocab_size: int
    :param vocab_size: the size of the vocabulary for this dataset. Defaults to
                       8000
    """
    global log
    global vocabulary
    global sentences
    global index_to_word
    global word_to_index
    global unknown
    global x_train
    global y_train

    log.info("Obtaining word frequency disribution.")
    word_freq = nltk.FreqDist(itertools.chain(*sentences))
    log.info("Found %d unique words." % len(word_freq.items()))

    vocabulary = word_freq.most_common(vocab_size - 1)

    log.info("Calculating percent of words captured...")
    total = 0
    for word in vocabulary:
        total += word_freq.freq(word[0])
    log.info("Percent of total words captured: %f" % total * 100)

    index_to_word = [word[0] for word in vocabulary]
    index_to_word.append(unknown)
    word_to_index = dict((word, index)
                        for index, word in enumerate(index_to_word))

    log.info("Replace all words not in vocabulary with unkown token.")
    for index, sentence in enumerate(sentences):
        sentences[index] = [word if word in word_to_index
                            else unknown for word in sentence]

    log.info("Creating training data.")
    x_train = np.asarray([[word_to_index[word] for word in sentence[:-1]]
                         for sentence in sentences])
    y_train = np.asarray([[word_to_index[word] for word in sentence[1:]]
                         for sentence in sentences])
# End of create_dataset()

def tokenize_paragraphs(num_examples=None):
    """
    Uses the nltk library to break comments down into paragraphs, and then
    tokenizes the words in the sentences. Also appends the paragraph start and
    end tokens to each paragraph.
    """
    global paragraphs
    global comments
    global paragraph_start
    global paragraph_end
    global log
    global space_token

    log.info("Breaking comments down into paragraphs.")
    for comment in comments:
        paragraphs.extend(re.split('\n+', comment.lower()))
    log.info("%d comments were broken down into %d paragraphs." %
             (len(comments), len(paragraphs)))

    log.info("Preprocessing paragraphs.")
    for item in paragraphs:
        item = item.replace(" ", " %s " % space_token)
        item = item.replace("\'\'", "\"")
        item = item.replace("``", "\"")

    for paragraph in paragraphs:
        if "[" in paragraph or "]" in paragraph:
            paragraphs.remove(paragraph)

    if (not num_examples is None) and num_examples < len(paragraphs):
        log.info("Reducing number of paragraphs to %d" % num_examples)
        paragraphs = paragraphs[:num_examples]

    log.info("Adding paragraph start and end tokens to paragraphs.")
    paragraphs = ["%s %s %s" % (paragraph_start, paragraph, paragraph_end)
                 for paragraph in paragraphs]

    log.info("Tokenizing words in paragraphs.")
    paragraphs = [nltk.word_tokenize(paragraph) for paragraph in paragraphs]
    paragraphs = list(paragraphs)
# End of tokenize_paragraphs()

def create_paragraph_dataset(vocab_size=8000):
    """
    Creates a dataset using the tokenized paragraphs.

    :type vocab_size: int
    :param vocab_size: the size of the vocabulary for this dataset. Defaults to
                       8000
    """
    global log
    global vocabulary
    global paragraphs
    global index_to_word
    global word_to_index
    global unknown
    global x_train
    global y_train

    log.info("Obtaining word frequency disribution.")
    word_freq = nltk.FreqDist(itertools.chain(*paragraphs))
    log.info("Found %d unique words." % len(word_freq.items()))

    vocabulary = word_freq.most_common(vocab_size - 1)

    log.info("Calculating percent of words captured...")
    total = 0
    for word in vocabulary:
        total += word_freq.freq(word[0])
    log.info("Percent of total words captured: %f" % total * 100)

    index_to_word = [word[0] for word in vocabulary]
    index_to_word.append(unknown)
    word_to_index = dict((word, index)
                        for index, word in enumerate(index_to_word))

    log.info("Replace all words not in vocabulary with unkown token.")
    for index, paragraph in enumerate(paragraphs):
        paragraphs[index] = [word if word in word_to_index
                            else unknown for word in paragraph]

    log.info("Creating training data.")
    x_train = np.asarray([[word_to_index[word] for word in paragraph[:-1]]
                         for paragraph in paragraphs])
    y_train = np.asarray([[word_to_index[word] for word in paragraph[1:]]
                         for paragraph in paragraphs])
# End of create_paragraph_dataset()

def tokenize_stories(num_examples=None):
    """
    Uses the nltk library to word tokenize entire comments, assuming that
    each comment is its own story. Also appends the story start and end tokens
    to each story.
    """
    global stories
    global comments
    global story_start
    global story_end
    global log
    global space_token
    global enter

    log.info("Retrieving stories from data.")
    stories = [comment.lower() for comment in comments]
    log.info("Found %d stories in the dataset." % len(stories))

    log.info("Preprocessing storeis")
    for item in stories:
        item = item.replace("\n", " %s " % enter)
        item = item.replace(" ", " %s " % space_token)
        item = item.replace("\'\'", "\"")
        item = item.replace("``", "\"")

    for story in stories:
        if "[" in story or "]" in story:
            stories.remove(story)

    if (not num_examples is None) and num_examples < len(stories):
        log.info("Reducing number of stories to %d" % num_examples)
        stories = stories[:num_examples]

    log.info("Adding story start and end tokens to stories.")
    stories = ["%s %s %s" % (story_start, story, story_end)
                 for story in stories]

    log.info("Tokenizing words in stories.")
    stories = [nltk.word_tokenize(story) for story in stories]
    stories = list(stories)
# End of tokenize_stories()

def create_story_dataset(vocab_size=8000):
    """
    Creates a dataset using the tokenized stories.

    :type vocab_size: int
    :param vocab_size: the size of the vocabulary for this dataset. Defaults to
                       8000
    """
    global log
    global vocabulary
    global stories
    global index_to_word
    global word_to_index
    global unknown
    global x_train
    global y_train

    log.info("Obtaining word frequency disribution.")
    word_freq = nltk.FreqDist(itertools.chain(*stories))
    log.info("Found %d unique words." % len(word_freq.items()))

    vocabulary = word_freq.most_common(vocab_size - 1)

    log.info("Calculating percent of words captured...")
    total = 0

    for word in vocabulary:
        total += word_freq.freq(word[0])

    log.info("Percent of total words captured: %f" % total * 100)

    index_to_word = [word[0] for word in vocabulary]
    index_to_word.append(unknown)
    word_to_index = dict((word, index)
                        for index, word in enumerate(index_to_word))

    log.info("Replace all words not in vocabulary with unkown token.")
    for index, story in enumerate(stories):
        stories[index] = [word if word in word_to_index
                            else unknown for word in story]

    log.info("Creating training data.")
    x_train = np.asarray([[word_to_index[word] for word in story[:-1]]
                         for story in stories])
    y_train = np.asarray([[word_to_index[word] for word in story[1:]]
                         for story in stories])
# End of create_story_dataset()

def save_dataset(path=None, filename=None):
    """
    Saves the created dataset to a specified file.

    :type path: string
    :param path: the path to the saved dataset file.
    """
    global x_train
    global y_train
    global index_to_word
    global word_to_index
    global vocabulary

    if path is None:
        path = input("Enter the path to the file where the dataset will"
                     " be stored: ")
    if filename is None:
        name = input("Enter the name of the file the dataset should be"
                     " saved as: ")

    setup.create_dir(path)
    with open(path + "/" + filename, "wb") as dataset_file:
        cPickle.dump((vocabulary, index_to_word, word_to_index, x_train,
                     y_train), dataset_file, protocol=2)
# End of save_dataset()

def load_dataset(path=None):
    """
    Loads a saved dataset so that it can be checked for correctness.

    :type path: string
    :param path: the path to the dataset
    """
    global log

    if path is None:
        path = input("Enter the path to the saved dataset: ")

    log.info("Loading saved dataset.")
    with open(path, "rb") as dataset_file:
        data = cPickle.load(dataset_file)
        vocabulary = data[0]
        index_to_word = data[1]
        word_to_index = data[2]
        x_train = data[3]
        y_train = data[4]

        log.info("Size of vocabulary is: %d" % len(vocabulary))
        log.info("Some words from vocabulary: %s" % index_to_word[:100])
        log.info("Number of examples: %d" % len(x_train))
        log.info("Sample training data: %s\n%s" % (x_train[:10], y_train[:10]))

        return data
# End of load_dataset()

def parse_arguments():
    """
    Parses command-line arguments and returns the array of arguments.

    :type return: list
    :param return: list of parsed command-line arguments
    """
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--vocab_size", default=8000, type=int,
                           help="The size of the dataset vocabulary.")
    arg_parse.add_argument("-c", "--num_comments", type=int,
                           help="The number of comments to be read.")
    arg_parse.add_argument("-n", "--num_examples", type=int,
                           help="The number of sentence examples to be saved.")
    arg_parse.add_argument("-s", "--source_path",
                           help="The source path to the data.")
    arg_parse.add_argument("-d", "--dest_path",
                           help="The destination path for the dataset.")
    arg_parse.add_argument("-f", "--dest_name",
                           help="The name of the dataset file.")
    arg_parse.add_argument("-t", "--source_type", default="csv",
                           help="The type of source data [currently only "
                                "the csv data size is supported].")
    arg_parse.add_argument("-p", "--log_dir", default='logging',
                           help="The logging directory.")
    arg_parse.add_argument("-l", "--log_name", default='DATA',
                           help="The name of the logger to be used.")
    arg_parse.add_argument("-e", "--test", action="store_true",
                           help="Specify if this is just a test run.")
    arg_parse.add_argument("-m", "--mode", default='sentences',
                           choices=['sentences', 'paragraphs', 'stories'],
                           help="Selects what constitutes an example in the "
                                "dataset.")
    return arg_parse.parse_args()
# End of parse_arguments()

if __name__ == "__main__":
    args = parse_arguments()
    set_up_logging(args.log_name, args.log_dir)
    if args.source_type == "csv":
        read_csv(args.source_path, args.num_comments)
    if args.mode == "sentences":
        tokenize_sentences(args.num_examples)
        create_sentence_dataset(args.vocab_size)
    if args.mode == "paragraphs":
        tokenize_paragraphs(args.num_examples)
        create_paragraph_dataset(args.vocab_size)
    if args.mode == "stories":
        tokenize_stories(args.num_examples)
        create_story_dataset(args.vocab_size)
    save_dataset(args.dest_path, args.dest_name)
    if args.test:
        load_dataset(args.dest_path + "/" + args.dest_name)
