# CPU = False

# # when running at lipml and not at slac, I need to pick my GPU:
# import os
# import subprocess
# x = str(subprocess.check_output(['hostname']))
# if "lipml" in x:
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1" if not CPU else "0"
    
import h5py
import numpy as np
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import pandas as pd
from pandas import HDFStore,DataFrame
#import tensorflow as tf
import time
import os

#Reading the dataframe
sample_name = "train_chunk_0_19999999" #Change here the dataframe 

subjets_df = pd.read_hdf('/lstore/calo/martafsilva/Xbb/Jet_color_ring/May19_Analysis/SamplesMay19_dataframes/%s.h5'%sample_name, key='table')



#Limiting the mass values for considering only values near the higgs
min_value = 110000
max_value = 140000

subjets_df = subjets_df.loc[(subjets_df['mass'] >= min_value) & (subjets_df['mass'] <= max_value)]


# Defining jet color ring 

def jet_color_ring_tt(row):
    
    # Create a dictionary for each subjet
    subjet1 = {"pt": row["pt_1"], "eta": row["eta_1"], "mass": row["mass_1"], "HadronConeExclTruthLabelID": row["HadronConeExclTruthLabelID_1"], "deta": row["deta_1"], "dphi": row["dphi_1"], "valid": row["valid_1"]}
    subjet2 = {"pt": row["pt_2"], "eta": row["eta_2"], "mass": row["mass_2"], "HadronConeExclTruthLabelID": row["HadronConeExclTruthLabelID_2"], "deta": row["deta_2"], "dphi": row["dphi_2"], "valid": row["valid_2"]}
    subjet3 = {"pt": row["pt_3"], "eta": row["eta_3"], "mass": row["mass_3"], "HadronConeExclTruthLabelID": row["HadronConeExclTruthLabelID_3"], "deta": row["deta_3"], "dphi": row["dphi_3"], "valid": row["valid_3"]}

    subjets = [subjet1, subjet2, subjet3]
    
    #filters the subjets that are not valid -> gives it Nan
    if not all([subjet["valid"] for subjet in subjets]):
        return np.nan
    
    #Identify which subjet corresponds to a, b and k
    # a and b -> analogous to the hard partons -> higher pt
    #k -> analogous to the soft gluon -> smaller pt
    
    a = max(subjets, key=lambda x: x["pt"])
    k = min(subjets, key=lambda x: x["pt"])
    
    #we will have only one element in the list so we select the first one
    b = [x for x in subjets if x != a and x != k][0]
    
    #Truth tagging - both a and b jets have to be b's
    if a['HadronConeExclTruthLabelID'] != 5 or b['HadronConeExclTruthLabelID'] != 5:
        return np.nan

    # Calculate the eta for each subjet difference
    deta_ab = a["deta"] - b["deta"]
    deta_ak = a["deta"] - k["deta"]
    deta_bk = b["deta"] - k["deta"]
    
    # Calculate the phi for each subjet difference
    phi_ab = a["dphi"] - b["dphi"]
    phi_ak = a["dphi"] - k["dphi"]
    phi_bk = b["dphi"] - k["dphi"]
    
    #Calculate the dR (theta)
    dR_ab = np.sqrt(deta_ab**2 + phi_ab**2 )
    dR_ak = np.sqrt(deta_ak**2 + phi_ak**2 )
    dR_bk = np.sqrt(deta_bk**2 + phi_bk**2 )
    
    if dR_ab < 0.01:
        return np.nan
    

    # Calculate the jet color ring
    jet_color_ring = (dR_ak**2 + dR_bk**2)/dR_ab**2
    return jet_color_ring



#subjets_df_JCR_tt = subjets_df.dropna().reset_index(drop=True)


# Apply the function to each row of the DataFrame
subjets_df["jet_color_ring_tt"] = subjets_df.apply(jet_color_ring_tt, axis=1)



#Defining Jet color ring variant 
def jet_color_ring_tt_l(row):
    
    # Create a dictionary for each subjet
    subjet1 = {"pt": row["pt_1"], "eta": row["eta_1"], "mass": row["mass_1"], "HadronConeExclTruthLabelID": row["HadronConeExclTruthLabelID_1"], "deta": row["deta_1"], "dphi": row["dphi_1"], "valid": row["valid_1"]}
    subjet2 = {"pt": row["pt_2"], "eta": row["eta_2"], "mass": row["mass_2"], "HadronConeExclTruthLabelID": row["HadronConeExclTruthLabelID_2"], "deta": row["deta_2"], "dphi": row["dphi_2"], "valid": row["valid_2"]}
    subjet3 = {"pt": row["pt_3"], "eta": row["eta_3"], "mass": row["mass_3"], "HadronConeExclTruthLabelID": row["HadronConeExclTruthLabelID_3"], "deta": row["deta_3"], "dphi": row["dphi_3"], "valid": row["valid_3"]}

    subjets = [subjet1, subjet2, subjet3]
    
    #filters the subjets that are not valid -> gives it Nan
    if not all([subjet["valid"] for subjet in subjets]):
        return np.nan
    
    #Identify which subjet corresponds to a, b and k
    # a and b -> analogous to the hard partons -> higher pt
    #k -> analogous to the soft gluon -> smaller pt
    
    a = max(subjets, key=lambda x: x["pt"])
    k = min(subjets, key=lambda x: x["pt"])
    
    #we will have only one element in the list so we select the first one
    b = [x for x in subjets if x != a and x != k][0]
    
    #Truth tagging - both a and b jets have to be b's
    if a['HadronConeExclTruthLabelID'] != 5 or b['HadronConeExclTruthLabelID'] != 5:
        return np.nan

    # Calculate the eta for each subjet difference
    deta_ab = a["deta"] - b["deta"]
    deta_ak = a["deta"] - k["deta"]
    deta_bk = b["deta"] - k["deta"]
    
    # Calculate the phi for each subjet difference
    phi_ab = a["dphi"] - b["dphi"]
    phi_ak = a["dphi"] - k["dphi"]
    phi_bk = b["dphi"] - k["dphi"]
    
    #Calculate the dR (theta)
    dR_ab = np.sqrt(deta_ab**2 + phi_ab**2 )
    dR_ak = np.sqrt(deta_ak**2 + phi_ak**2 )
    dR_bk = np.sqrt(deta_bk**2 + phi_bk**2 )
    
    if dR_ab < 0.01:
        return np.nan
    

    # Calculate the jet color ring
    jet_color_ring_l = dR_ak**2 + dR_bk**2-dR_ab**2
    return jet_color_ring_l


#subjets_df_JCR_tt = subjets_df_JCR_tt.dropna().reset_index(drop=True)


# Apply the function to each row of the DataFrame
subjets_df["jet_color_ring_tt_l"] = subjets_df.apply(jet_color_ring_tt_l, axis=1)




#Defining XbbScore

def XbbScore(row):
    return np.log(row['Xbb2020v3_Higgs']/row['Xbb2020v3_QCD'])

subjets_df["XbbScore"] = subjets_df.apply(XbbScore, axis=1)
#subjets_df_JCR = subjets_df_JCR.dropna().reset_index(drop=True)

#Saving the new dataframe with JCR and XbbScore values 

subjets_df.to_hdf('df_JCR_tt%s.h5'%sample_name, key='table', mode='w')

print("the table is done")