"""
Contains constants for use within the project.

Copyright (c) 2017 Frank Derry Wanye

Date: 9 September, 2017
"""
##########################################
# TOKENS
##########################################
UNKNOWN = "UNKNOWN_TOKEN"
PAD = "PADDING_TOKEN"
SENT_START = "SENTENCE_START"
SENT_END = "SENTENCE_END"
PARA_START = "PARAGRAPH_START"
PARA_END = "PARAGRAPH_END"
STORY_START = "STORY_START"
STORY_END = "STORY_END"
CARRIAGE_RETURN = "CARRIAGE_RETURN"
SPACE = "SPACE_TOKEN"

#########################################
# LOGGING
########################################
LOG_NAME = "TERRY"

#########################################
# DEFAULT DIRECTORIES
#########################################
MODEL_DIR = "models/"
DATASETS_DIR = "datasets/"
TENSORBOARD = "tensorboard/"
LOG_DIR = "logging/"
RAW_DATA_DIR = "raw_data/"

#########################################
# VARIABLE SCOPES FOR TENSORBOARD
#########################################
EMBEDDING = "embedding_layer"
HIDDEN = "hidden_layers"
OUTPUT = "output_layer"
TRAINING = "network_training"

#########################################
# TENSORBOARD SUMMARY NAMES
#########################################


#########################################
# DEFAULT FILENAMES
#########################################
LATEST_WEIGHTS = "latest_weights.pkl"
WEIGHTS = "/weights.pkl"
PLOT = "/loss_plot.png"
META = "meta.pkl"
LOGGING = "logging.log"
