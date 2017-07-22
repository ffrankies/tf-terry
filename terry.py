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
# Provides an interface for generating sentences, paragraphs and stories
# using already generated models.
#
# Future plan is to have it call functions from rnn.py using arguments saved
# in a yaml or json file.
###############################################################################
import data_utils # Creating datasets
import rnn # The base neural network
import argparse # The argument parser
import json # For passing output to php files
import logging
import logging.handlers
import os

log = logging.getLogger("TERRY")
log.setLevel(logging.DEBUG)

RNN=None

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
            if os.path.dirname(dirPath) != "":
                os.makedirs(os.path.dirname(dirPath))
        except OSError as exception:
            if exception.errno != 17:
                raise
# End of createDir()

def set_up_logging(path='logging/terry.log'):
    """
    Sets up logging for the data formatting.

    :type name: String
    :param name: the name of the logger. Defaults to 'DATA'

    :type dir: String
    :param dir: the directory in which the logging will be done. Defaults to
                'logging'
    """
    createDir(path) # Create log directory in system if it isn't already there
    global log

    handler = logging.handlers.RotatingFileHandler(
        filename=path,
        maxBytes=1024*512,
        backupCount=5
    )

    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(message)s"
    )

    handler.setFormatter(formatter)

    log.addHandler(handler)
# End of set_up_logging()

def parse_arguments():
    """
    Parses the command line arguments and returns the namespace with those
    arguments.

    :type return: Namespace object
    :param return: The Namespace object containing the values of all supported
                   command-line arguments.
    """
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-l", "--log", default="./logging/terry.log",
                           help="The path to the file to be used for storing "
                                "logs.")
    arg_parse.add_argument("-s", "--sentence", action="store_true",
                           help="Used to output sentences.")
    arg_parse.add_argument("-p", "--paragraph", action="store_true",
                           help="Used to output paragraphs.")
    arg_parse.add_argument("-t", "--story", action="store_true",
                           help="Used to output stories.")
    arg_parse.add_argument("-n", "--num", default=1, type=int,
                           help="The number of outputs to produce.")
    arg_parse.add_argument("-m", "--min_length", default=0, type=int,
                           help="The minimum length of the output to produce.")
    return arg_parse.parse_args()
# End of parse_arguments()

def generate_sentences(min_length=0, num=1):
    """
    Generates 'num' sentences of minimum length 'min_length', and returns a
    json object with these sentences.

    :type min_length: int
    :param min_length: the minimum length of the sentences to be produced.

    :type num: int
    :param num: the number of sentences to be produced.
    """
    global log
    global RNN
    log.info("Generating %d sentences of minimum length %d." %
             (num, min_length))
    RNN = rnn.GruRNN(dataset="datasets/all_story_sents.pkl", trunc=100,
                     model="models/sentences.pkl")
    sents = [RNN.pretty_print(RNN.generate_sentence()[0]) for i in range(num)]
    log.info("Generated sentence(s): %s" % sents)
    return json.dumps(",".join(sents))
# End of generate_sentences()

def generate_paragraphs(min_length=0, num=1):
    """
    Generates 'num' paragraphs of minimum length 'min_length', and returns a
    json object with these paragraphs.

    :type min_length: int
    :param min_length: the minimum length of the paragraphs to be produced.

    :type num: int
    :param num: the number of paragraphs to be produced.
    """
    global log
    global RNN
    log.info("Generating %d paragraphs of minimum length %d." %
             (num, min_length))
    RNN = rnn.GruRNN(dataset="datasets/all_paragraphs.pkl", trunc=500,
                     model="models/paragraphs.pkl")
    paras = [RNN.pretty_print(RNN.generate_paragraph()[0]) for i in range(num)]
    log.info("Generated paragraph(s): %s" % paras)
    return json.dumps(",".join(paras))
# End of generate_sentences()

def generate_stories(min_length=0, num=1):
    """
    Generates 'num' stories of minimum length 'min_length', and returns a
    json object with these stories.

    :type min_length: int
    :param min_length: the minimum length of the stories to be produced.

    :type num: int
    :param num: the number of stories to be produced.
    """
    global log
    global RNN
    log.info("Generating %d stories of minimum length %d." %
             (num, min_length))
    log.info("Setting up the RNN")
    RNN = rnn.GruRNN(dataset="datasets/all_stories.pkl", trunc=500,
                     model="models/mid_story.pkl")
    log.info("Generating the stories")
    stories = [RNN.pretty_print(RNN.generate_story(min_length=min_length)[0])
               for i in range(num)]
    log.info("Generated story(ies): %s" % stories)
    return json.dumps(",".join(stories))
# End of generate_sentences()

if __name__ == "__main__":
    args = parse_arguments()
    set_up_logging(args.log)
    if args.sentence:
        print(generate_sentences(args.min_length, args.num))
    elif args.paragraph:
        print(generate_paragraphs(args.min_length, args.num))
    elif args.story:
        print(generate_stories(args.min_length, args.num))
