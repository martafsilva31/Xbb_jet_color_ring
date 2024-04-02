import json
import os
import subprocess
from model_training import train_model
import tensorflow as tf

# Specify the path to the configuration file
config_file = "config.json"

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')

# Load the configuration from the JSON file
with open(config_file) as f:
    config = json.load(f)

# Iterate over each feature set in the configuration
for feature_set in config['train_features']:
    # Get the training features for the current feature set
    train_features = feature_set['features']

    # Get the model name for the current feature set
    model_name = feature_set['model_name']

    # Train the model for the current feature set
    train_model(config_file, train_features, model_name)

