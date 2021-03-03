#########################################################################
# Description:    Script to determin optimal prediction thresholds	    #
#                 in order to use samples for learning.			            #
# Authors:        Matthias Niggli/CIEB UniBasel                         #
# Last Revised:   03.03.2021                                            #
#########################################################################

##################################################
###### Load packages and set directories #########
##################################################

#### Import packages ---------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
print("All packages loaded.")

import joblib

#### Set directory ---------------------------------------------------------
# path = "C:/Users/Matthias/Documents/GithubRepos/name_to_origin"
path = "/scicore/home/weder/nigmat01/name_to_origin"
print("Directories specified")

##################################################
###### Load data and trained RF model ############
##################################################

# model
rf = joblib.load(path + "/Classification_models/rf_origin_assignment"+ ".joblib")

# data
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
print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

###############################################
###### Subsampling to increase accuracy #######
###############################################

max_proba = pd.DataFrame(rf.predict_proba(x_test)).max(axis = 1)
print("Mean of highest class probability is: ", round(100 * max_proba.mean(), 1), "%") # 78.8%
print("Minimum of highest class probability is: ", round(100 * max_proba.min(), 1), "%") # 11.8%
print("Median of highest class probability is: ", round(100 * max_proba.median(), 1), "%") # 88.3%

THRESHOLD = [x / 100 for x in range(40, 75, 5)]
ACCURACIES = []
SAMPLE_FRACTION = []
for THRES in THRESHOLD:
    y_pred_thres = rf_save.predict(x_test)
    y_pred_thres = y_pred_thres[max_proba > THRES]
    y_test_thres = y_test[max_proba > THRES]
    
    thres_acc = metrics.accuracy_score(y_test_thres, y_pred_thres)
    sample_fraction = len(y_test_thres) / len(y_test)
    
    print("Threshold of", THRES, "drops", round(100 * (1 - sample_fraction), 1), "% of samples.") #22.9%
    print("Overall accuracy with probability threshold of", THRES, "is", round(thres_acc * 100, 1), "%")
    
    ACCURACIES.append(thres_acc)
    SAMPLE_FRACTION.append(sample_fraction)

# baseline model accuarcy
y_pred = rf.predict(x_test)
ACC = metrics.accuracy_score(y_test, y_pred)

eval_df = pd.DataFrame({"Threshold": ["No"] + THRESHOLD,
              "Accuracy": [ACC] + ACCURACIES, 
              "Sample_Fraction": [1] + SAMPLE_FRACTION})
print(eval_df)




##########################################
###### Get test sample of names ##########
##########################################

# Question: Either athletes or inventors or both?

N = 8000

# depending on choice: assign origins based on rf classifier

# tokenize features

##############################################################
###### Use remaining samples to train a simple LSTM ##########
##############################################################

max_proba = pd.DataFrame(rf.predict_proba(x_test)).max(axis = 1)
print("Mean of highest class probability is: ", round(100 * max_proba.mean(), 1), "%") # 78.8%
print("Minimum of highest class probability is: ", round(100 * max_proba.min(), 1), "%") # 11.8%
print("Median of highest class probability is: ", round(100 * max_proba.median(), 1), "%") # 88.3%

THRESHOLD = 0.6
sub_res = res.iloc[np.where(max_proba >= THRESHOLD)]
print("Drops ", round(100 * (1 - len(sub_res) / len(res)), 1), "% of samples.") #22.9%
acc = sum(sub_res["match"]) / len(sub_res)
print("Overall accuracy of the Random Forest classifier is ", round(acc * 100, 1), "%") 

# subsample remaining data according to some threshold for training

# tokenize

# train simple LSTM

# evaluate on test set

# compare LSTM performances for different subsetting thresholds

# choose best threshold


### TO DO'S
# 3) try out different threshols for clearly predicted names. the goal is to have more than 95% accuracy
# 4) save model and apply to patent inventors.



