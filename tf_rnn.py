"""
This project is licensed under the MIT license:
https://github.com/ffrankies/Terry/blob/master/LICENSE

This project uses the Theano library:
https://arxiv.org/pdf/1605.02688.pdf

This project is based on the following tutorial:
https://goo.gl/DPf37h

Copyright (c) 2017 Frank Derry Wanye

Date: 17 July, 2017
"""

#########################################################################################
# A simple RNN based on https://www.tensorflow.org/tutorials/recurrent
########################################################################################
import tensorflow as tf
from utils import setup
from utils.model import RNNModel
from utils import datasets
from utils.trainer import train
from utils.generator import generate_output
from utils import saver

# settings = setup.parse_arguments()
#
# # TO-DO: import data here
#
# # RNN code here
# rnn = RNN(settings)
# rnn.train()

if __name__ == "__main__":
    settings = setup.parse_arguments()
    # rnn = RNNModel(settings)
    # train(rnn, rnn.session, False)
    # generate_output(rnn)
    # saver.save_model(rnn)
    # rnn.session.close()

    # print("\n\n\nSaved with pickle. Loading with pickle next...\n\n\n")
    # tf.reset_default_graph()

    rnn2 = RNNModel(settings)
    saver.load_model(rnn2, "pickle_model")
    generate_output(rnn2)
    train(rnn2)
    generate_output(rnn2)
    rnn2.session.close()