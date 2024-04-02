# Packages needed 

import h5py
import json
import numpy as np
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.metrics import roc_curve, auc
from pandas import HDFStore,DataFrame
import tensorflow as tf
import time
import os
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from preprocess import preprocess_data
np.random.seed(1234) # set the np random seed for reproducibility


# Specify the path to the configuration file
config_file = "config.json"

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[1], 'GPU')

# Load the configuration from the JSON file
with open(config_file) as f:
    config = json.load(f)

    # Set the sample name
    sample_name = config.get('sample_name')

plt.figure(figsize = (10,6))

# Iterate over each feature set in the configuration
for feature_set in config['train_features']:
    # Get the training features for the current feature set
    train_features = feature_set['features']

    # Get the model name for the current feature set
    model_name = feature_set['model_name']

    # Load the training, validation, and test data
    df_train = pd.read_hdf(config.get('train_data_path'), key='table')
    df_val = pd.read_hdf(config.get('val_data_path'), key='table')
    df_test = pd.read_hdf(config.get('test_data_path'), key='table')

    # Preprocess the data
    X_train, Y_train_cat, X_val, Y_val_cat, X_test, Y_test_cat, input_shape = preprocess_data(df_train, df_val, df_test, train_features)
    
    model_path = "/lstore/calo/martafsilva/Xbb/xbb-nn/May19_tt"

    #Import the model
    model = tf.keras.models.load_model(f"{model_path}/{sample_name}_{model_name}/{sample_name}_{model_name}")
    
    #Make predictions
    y_pred_model = model.predict(X_test, use_multiprocessing=True,batch_size=1024)
    
    #Calculating the Xbbscore for the 3 datasets

    XbbScore = np.log(y_pred_model[:,0]/y_pred_model[:,3])

    df_test['XbbScore1'] = XbbScore.tolist()


    #Reducing the dataframes
    df_test_r = df_test[(df_test["flavour_label"] == 0) | (df_test["flavour_label"] == 3)]
    df_test_r  = df_test_r.reset_index(drop=True)
    
    df_test_r['signal'] = df_test_r['flavour_label'].replace({0: 1, 3: 0})

    #Plotting the ROCs

    # compute the ROC curve for xbbscore_cut
    fpr_xbbscore1, tpr_xbbscore1, thresholds_xbbscore1 = roc_curve(df_test_r['signal'], df_test_r['XbbScore1'])
    roc_auc_xbbscore1 = auc(fpr_xbbscore1, tpr_xbbscore1)

    # # plot ROC curve -> choose the style 
    #plt.plot(fpr_xbbscore1, tpr_xbbscore1, label='XbbScore with ' + str(model_name)+   ' (area = %0.2f)'% roc_auc_xbbscore1)
    
    # plot ROC curve HEP STYLE 
    plt.plot(tpr_xbbscore1[fpr_xbbscore1 != 0], 1 / fpr_xbbscore1[fpr_xbbscore1 != 0], label='XbbScore with ' + str(model_name))

#Here chose the style again
# fpr_xbbscoret, tpr_xbbscoret, thresholds_xbbscoret = roc_curve(df_test_r['signal'], df_test_r['XbbScore'])
# roc_auc_xbbscoret = auc(fpr_xbbscoret, tpr_xbbscoret)
# plt.plot(fpr_xbbscoret, tpr_xbbscoret, label='XbbScore from data (area = %0.2f)'% roc_auc_xbbscoret)

# plt.xlabel('False Positive Rate', fontsize = 14)
# plt.ylabel('True Positive Rate', fontsize = 14)
# plt.legend(loc="lower right")
# plt.xticks(fontsize = 12)
# plt.yticks(fontsize = 12)
# plt.title('ROC without weights')
# plt.savefig("ROCs_"+ str(sample_name)+".jpeg")


#HEP STYLE
#plt.plot(tpr_xbbscoret[fpr_xbbscoret != 0], 1 / fpr_xbbscoret[fpr_xbbscoret != 0], label='XbbScore from data')


plt.xlabel('Signal Efficiency', fontsize = 14)
plt.ylabel('Background Rejection', fontsize = 14)
plt.legend(loc="lower left", fontsize=12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.yscale('log')
plt.xlim(0.4,1.02)
plt.ylim(1,1000)
plt.title('ROC without weights')
plt.savefig("ROC_curve_HEP" + str(sample_name)+ ".jpeg")


################################ TO INCLUDE WEIGHTS ####################

#substitute with these 

# fpr_xbbscore1, tpr_xbbscore1, thresholds_xbbscore1 = roc_curve(df_test_r['signal'], df_test_r['XbbScore1'],sample_weight=df_test_r['mcEventWeight'].values)
# roc_auc_xbbscore1 = auc(fpr_xbbscore1, tpr_xbbscore1)