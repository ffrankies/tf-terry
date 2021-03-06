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
from tf_rnn import setup
from tf_rnn.model import RNNModel
from tf_rnn import datasets
from tf_rnn.trainer import train
from tf_rnn.generator import generate_output
from tf_rnn import saver

if __name__ == "__main__":
    rnn = RNNModel()
    train(rnn)
    #    generate_output(rnn)
    saver.save_model(rnn)
    rnn.session.close()

    # print("\n\n\nSaved with pickle. Loading with pickle next...\n\n\n")
    # tf.reset_default_graph()

    # rnn2 = RNNModel(settings)
    # saver.load_model(rnn2, "test_model")
    # # generate_output(rnn2)
    # train(rnn2)
    # generate_output(rnn2)
    # saver.save_model(rnn2)
    # rnn2.session.close()
