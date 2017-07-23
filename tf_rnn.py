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
from utils import setup
from utils.rnn import RNN
from utils import datasets

# settings = setup.parse_arguments()
#
# # TO-DO: import data here
#
# # RNN code here
# rnn = RNN(settings)
# rnn.train()

if __name__ == "__main__":
    settings = setup.parse_arguments()
    rnn = RNN(settings)
    rnn.train()
    rnn.generate_output()
