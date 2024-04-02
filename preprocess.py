import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler

def preprocess_data(df_train, df_val, df_test, trainFeatures):
    X_train = df_train[trainFeatures]
    X_val = df_val[trainFeatures]
    X_test = df_test[trainFeatures]

    Y_train = df_train["flavour_label"]
    Y_val = df_val["flavour_label"]
    Y_test = df_test["flavour_label"]

    # Convert labels to categorical data
    Y_train_cat = to_categorical(Y_train)
    Y_val_cat = to_categorical(Y_val)
    Y_test_cat = to_categorical(Y_test)

    # Mean imputation
    mean_vector = X_train.mean()
    X_train.fillna(mean_vector, inplace=True)
    X_test.fillna(mean_vector, inplace=True)
    X_val.fillna(mean_vector, inplace=True)

    # Data standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    # Determine input shape based on the length of training features
    input_shape = X_train.shape[1]

    return X_train, Y_train_cat, X_val, Y_val_cat, X_test, Y_test_cat, input_shape
