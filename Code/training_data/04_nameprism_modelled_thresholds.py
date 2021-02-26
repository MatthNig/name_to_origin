#########################################################################
# Description:    Script to assign NamePrism Leaf Nationalities to      #
#                 ethnic origins.                                       #
# Authors:        Matthias Niggli/CIEB UniBasel                         #
# Date:           25.02.2021                                            #
#########################################################################

#######################################
###### Load packages and data #########
#######################################

#### Import packages ---------------------------------------------------------
import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


import os

# import random
# import sys
# from matplotlib import pyplot as plt
print("All packages loaded.")

#### Set directory
path = "C:/Users/Matthias/Documents/GithubRepos/name_to_origin"
#path = "/scicore/home/weder/nigmat01/name_to_origin"
os.chdir(path)
print("Directories specified")

#### Load & process the data
df = pd.read_csv("Data/API_verification_sample.csv")
df = df.drop(labels = ["Name", "Year", "full_name_encoded"], axis = 1)

#######################################
###### Encode ethnic origins ##########
#######################################

le = preprocessing.LabelEncoder()
le.fit(df["origin"])
df["origin_encoded"] = le.fit_transform(df["origin"])

response = np.array(df["origin_encoded"])
features = np.array(df.drop(["origin", "origin_encoded"], axis = 1))

################################################
###### Split to training and test set ##########
################################################

x_train, x_test, y_train, y_test = train_test_split(features, response, test_size = 0.2, random_state = 25022021)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)


#################################
###### Define Baseline ##########
#################################

# create a reference dictionary that maps ethnic origins to NamePrism Leaf Nationalities
origin_dict = df[["origin_encoded", "origin"]].drop_duplicates()

leaf_nationalities = pd.Series(df.columns[1:-1])
leaf_nationalities = pd.DataFrame(leaf_nationalities, columns = ["leaf_nationality"])
leaf_nationalities["origin"] = ["Balkans", "Arabic", "Italian", "EastEurope",
                              np.nan, np.nan, np.nan, "French",
                              "SouthEastAsia", "SouthEastAsia", np.nan, np.nan,
                              "SouthEastAsia", "Scandinavian", "SouthEastAsia", "Scandinavian",
                              "Persian", "Scandinavian", "Arabic", np.nan, 
                              "Arabic", "Portugese", "Slawic", "Arabic",
                              np.nan, "Japan", "German", "China",
                              "India", "Hispanic", "Scandinavian", "Turkey",
                              np.nan, "AngloSaxon", "SouthEastAsia", "Korea",
                              np.nan, "EastEurope", "SouthEastAsia"]
leaf_nationalities["feature_idx"] = pd.Series(range(39))

origin_dict = pd.merge(origin_dict, leaf_nationalities, how = "left", on = ["origin"])

# retrieve highest origin probabilities in test datat   
pred_response_baseline = pd.DataFrame(x_test).idxmax(axis = 1)
pred_response_baseline = pd.DataFrame(pred_response_baseline, columns=["feature_idx"])
pred_response_baseline = pd.merge(pred_response_baseline, origin_dict, how = "left", on = ["feature_idx"])
pred_response_baseline = pred_response_baseline.rename(columns={"origin_encoded": "predicted_origin_encoded", 
                                                                "origin": "predicted_origin"}) 
pred_response_baseline = pred_response_baseline.loc[:, ['predicted_origin_encoded', 'predicted_origin']]

# get true responses in test data
pred_response_baseline["true_origin"] = y_test

# evaluate
pred_response_baseline["match"] = np.where(
    pred_response_baseline['predicted_origin_encoded'] == pred_response_baseline["true_origin"], 1, 0)
acc = len(pred_response_baseline[(pred_response_baseline.match == 1)]) / len(pred_response_baseline)
print("Accuracy when classifying to the highest origin prediction is ", round(acc * 100, 1), "%")


#################################
###### Train RandomForest #######
#################################


