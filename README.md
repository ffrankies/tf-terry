[![codebeat badge](https://codebeat.co/badges/4a72f05a-6a47-4494-aa74-9e1728bc7339)](https://codebeat.co/projects/github-com-ffrankies-tf-terry-master)

# Terry
[Terry](https://github.com/ffrankies/terry) is a digital writer made with a Recurrent Neural Network.
tf-terry

tf-Terry (this project) is a tensorflow port of the same project. Made as a starting point for the Kindschi Fellowship.

<!-- TOC -->

- [Journal](#journal)
- [Functionality](#functionality)
- [Problems](#problems)

<!-- /TOC -->

## Journal

A [journal](https://github.com/ffrankies/kindschi-fellowship-journal/blob/master/README.md) detailing my weekly work process. Since this project is done as a part of the Kindschi Fellowship, it will contain information relevant to both this project and the main research project.

## Functionality

- Trains a Recurrent Neural Network on text input
- Saves tensorflow event logs so that they can be viewed with tensorboard
- Saves tensorflow variable weights so that they can be recovered at a later time
- Loads tensorflow variable weights so that they can be reused

## Problems

- Loading a pre-trained RNN with the built-in tensorflow Saver() object does not work.
  [Related issue](https://github.com/tensorflow/tensorflow/issues/6683)