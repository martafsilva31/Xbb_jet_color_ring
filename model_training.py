import os
import json
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess import preprocess_data
from model_architecture import build_model

def train_model(config_file, training_features, model_name):
    # Load the configuration from the JSON file
    with open(config_file) as f:
        config = json.load(f)

    # Set the sample name
    sample_name = config.get('sample_name')

    # Load the model parameters
    dense_units = config['model_params']['dense_units']
    num_classes = config['model_params']['num_classes']
    activation = config['model_params']['activation']
    batch_normalization = config['model_params']['batch_normalization']

    # Load the training parameters
    optimizer = config['training_params']['optimizer']
    learning_rate = config['training_params']['learning_rate']
    decay = config['training_params']['decay']
    batch_size = config['training_params']['batch_size']
    epochs = config['training_params']['epochs']

    # Load the training, validation, and test data
    df_train = pd.read_hdf(config.get('train_data_path'), key='table')
    df_val = pd.read_hdf(config.get('val_data_path'), key='table')
    df_test = pd.read_hdf(config.get('test_data_path'), key='table')

    # Preprocess the data
    X_train, Y_train_cat, X_val, Y_val_cat, X_test, Y_test_cat, input_shape = preprocess_data(df_train, df_val, df_test, training_features)

    # Create the model
    model = build_model(input_shape)

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, Y_train_cat, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val_cat))

    # Create a new folder for the model
    output_folder = f"{sample_name}_{model_name}"
    os.makedirs(output_folder, exist_ok=True)

    # Save the trained model
    save_model_path = os.path.join(output_folder, f"{sample_name}_{model_name}")
    model.save(save_model_path)

    # Generate loss and accuracy plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # Save the plots in the same directory as the model
    loss_plot_path = os.path.join(output_folder, "loss_plot.jpeg")
    

    plt.savefig(loss_plot_path)
    

    plt.show()