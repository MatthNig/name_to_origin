#########################################################################
# Description:    Script to construct the inventor sample for training  #
#                 unsing prediction chosen probability thresholds	    #
# Authors:        Matthias Niggli/CIEB UniBasel                         #
# Last Revised:   12.03.2021                                            #
#########################################################################

##################################################
###### Load packages and set directories #########
##################################################

#### Import packages ---------------------------------------------------------
import numpy as np
import pandas as pd
import pyreadr

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

import xgboost as xgb

#### Set directory ---------------------------------------------------------
import os
path = "C:/Users/Matthias/Documents/GithubRepos/name_to_origin"
if os.path.isdir(path) == False:
    path = "/scicore/home/weder/nigmat01/name_to_origin"
if os.path.isdir(path):
    print("Directories specified")
else: print("Could not find directory")

DatDir = "/scicore/home/weder/GROUP/Innovation/01_patent_data/created data"

########################
###### XGBoost #########
########################

#### load training data -------------------------------------------------------
df = pd.read_csv(path+"/Data/nameprism_stratified_sample.csv")
df = df.drop(labels = ["Name", "Year", "full_name_encoded"], axis = 1)

le = preprocessing.LabelEncoder()
le.fit(df["origin"])
df["origin_encoded"] = le.fit_transform(df["origin"])

response = np.array(df["origin_encoded"])
features = np.array(df.drop(["origin", "origin_encoded"], axis = 1))

x_train, x_test, y_train, y_test = train_test_split(features, response, 
                                                    test_size = 0.2, 
                                                    random_state = 25022021)

#### train the model using the best parameter values --------------------------------
xgb_model = xgb.XGBClassifier(random_state = 8032021, n_estimators = 140,
                            learning_rate = 0.1, min_child_weight = 3, max_depth = 3)
xgb_model.fit(X = x_train, y = y_train)
y_pred = xgb_model.predict(x_test)
xgb_base_acc = metrics.accuracy_score(y_test, y_pred)
print("Overall accuracy of the XGBoost is ", 
      round(xgb_base_acc * 100, 1), "%") # 85.2%
f1 = metrics.f1_score(y_true = y_test, y_pred = y_pred, average = "weighted")
print("Weighted F1 score of XGBoost is: ", 
      round(f1 * 100, 1), "%") # 85.1%


##############################
###### Inventor Data #########
##############################

df = pyreadr.read_r(DatDir+"/origin_training_sample.rds")
df = df[None]

xgb_model.predict_proba(df[])