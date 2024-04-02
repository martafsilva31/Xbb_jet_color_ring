import tensorflow as tf

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(250, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(250, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(250, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(250, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(250, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(250, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    return model
