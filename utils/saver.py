"""
Provides an interface for saving and loading various aspects of the 
tensorflow model to file.

Copyright (c) 2017 Frank Derry Wanye

Date: 6 September, 2017
"""

import tensorflow as tf
import time
import pickle

from .constants import MODEL_DIR
from . import setup

def create_model_dir(model_settings):
    """
    Creates the directory in which to save the model.

    :type model_settings: Namespace()
    :param model_settings: the model's settings.

    :type return: String
    :param return: the path to the created directory
    """
    model_name = model_settings.model_name
    if model_name is None:
        model_path = MODEL_DIR + time.strftime("%d%m%y%H") + "/"
    else:
        model_path = MODEL_DIR + model_name + "/"
    setup.create_dir(model_path)
    return model_path
# End of create_model_dir()

def save_model(model):
    """
    Save the current model's weights in the models/ directory.

    :type model: RNNModel()
    :param model: The model to save.
    """
    weights = model.variables.get_weights()
    with open(model.model_path + "/weights.pkl", "wb") as weights_file:
        pickle.dump(weights, weights_file)
# End of save_model()

def load_model(model, model_name_or_timestamp):
    """
    Load the model contained in the given directory.

    :type model: RNNModel()
    :param model: The model to which to restore previous data.

    :type model_name_or_timestamp: String
    :param model_name_or_timestamp: the name or timestamp of the model to load
    """
    weights_path = MODEL_DIR + model_name_or_timestamp + "/weights.pkl"
    with open(weights_path, "rb") as weights_file:
        weights = pickle.load(weights_file)
        model.variables.set_weights(weights)
# End of load_model()
