[![codebeat badge](https://codebeat.co/badges/4a72f05a-6a47-4494-aa74-9e1728bc7339)](https://codebeat.co/projects/github-com-ffrankies-tf-terry-master)

# Terry
[Terry](https://github.com/ffrankies/terry) is a digital writer made with a Recurrent Neural Network.
tf-terry

tf-Terry (this project) is a tensorflow port of the same project. Made as a starting point for the Kindschi Fellowship.

<!-- TOC -->

- [Journal](#journal)
- [Environment](#environment)
- [Directory structure](#directory-structure)
- [Functionality](#functionality)
- [Problems](#problems)

<!-- /TOC -->

## Journal

A [journal](https://github.com/ffrankies/kindschi-fellowship-journal/blob/master/README.md) detailing my weekly work process. Since this project is done as a part of the Kindschi Fellowship, it will contain information relevant to both this project and the main research project.

## Environment

- Bash on Ubuntu on Windows
- Python v3.5.2
- tensorflow v1.3.0 (the latest version of tensorboard will only work with tensorflow > v1.2.5)
- tensorflow-tensorboard v0.1.6
- matplotlib v2.0.2
- nltk v3.2.4

## Directory structure

Directory/file                             | Contains
------------------------------------------:|:------------------------------------------
datasets                                   | Pre-processed datasets, ready for training
models                                     | Contains saved weights for pre-trained models, as well as logs and metadata.
models/[model_name]/latest_weights.pkl     | Contains the weights from the latest run.
models/[model_name]/logging.log            | Consolidated log file for all model runs.
models/[model_name]/meta.pkl               | Model metadata, used when loading a pre-trained model.
models/[model_name]/latest_output.txt      | Sample output from the latest run.
models/[model_name]/tensorboard/           | Contains tensorboard event logs.
models/[model_name]/run_[num]/             | Contains data for a given run.
models/[model_name]/run_[num]/loss_plot.png | A graph of the losses for the given run.
models/[model_name]/run_[num]/weights.pkl  | The last saved weights from the given run.
models/[model_name]/run_[num]/output.txt   | Sample output from the given run. 

## Functionality

- Trains a Recurrent Neural Network on text input
- Saves tensorflow event logs so that they can be viewed with tensorboard
- Organizes tensorflow sessions in 'runs'
- Saves tensorflow variable weights so that they can be recovered at a later time
- Loads tensorflow variable weights so that they can be reused

## Problems

- Loading a pre-trained RNN with the built-in tensorflow Saver() object does not work.
  [Related issue](https://github.com/tensorflow/tensorflow/issues/6683)