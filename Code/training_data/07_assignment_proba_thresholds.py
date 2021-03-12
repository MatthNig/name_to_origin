#########################################################################
# Description:    Script to determine optimal prediction thresholds	    #
#                 in order to use samples for learning.		            #
# Authors:        Matthias Niggli/CIEB UniBasel                         #
# Last Revised:   11.03.2021                                            #
#########################################################################

##################################################
###### Load packages and set directories #########
##################################################

#### Import packages ---------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
print("All packages loaded.")

#### Set directory ---------------------------------------------------------
import os
path = "C:/Users/Matthias/Documents/GithubRepos/name_to_origin"
if os.path.isdir(path) == False:
    path = "/scicore/home/weder/nigmat01/name_to_origin"
if os.path.isdir(path):
    print("Directories specified")
else: print("Could not find directory")

##################################################
###### Load data and trained RF model ############
##################################################

# data
df = pd.read_csv(path+"/Data/nameprism_stratified_sample.csv")
le = preprocessing.LabelEncoder()
le.fit(df["origin"])
df["origin_encoded"] = le.fit_transform(df["origin"])
response = np.array(df["origin_encoded"])
features = np.array(df.drop(["Name", "Year", "full_name_encoded", "origin", "origin_encoded"], axis = 1))
indices = np.arange(len(df))

# use only those samples the XGB did not use for training:
x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
    features, response, indices,
    test_size = 0.2, random_state = 25022021)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

##############################################################################################
###### Subsampling based on highest class probability to increase accuracy ###################
##############################################################################################

# best parameters from hyperparameter tuning the XGB:
# {'n_estimators': 140, 'min_child_weight': 3, 'max_depth': 3, 'learning_rate': 0.1}
xgb_model = xgb.XGBClassifier(random_state = 8032021, n_estimators = 140,
                            learning_rate = 0.1, min_child_weight = 3, max_depth = 3)
xgb_model.fit(X = x_train, y = y_train)

y_pred = xgb_model.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_true = y_test, y_pred = y_pred, average = "weighted")
print("Overall accuracy of the tuned XGBoost is ", 
      round(acc * 100, 1), "%") # 85.2%
print("Weighted F1 score of the tuned XGBoost is: ", 
      round(f1 * 100, 1), "%") # 85.1%

max_proba = pd.DataFrame(xgb_model.predict_proba(x_test)).max(axis = 1)
print("Mean of highest class probability is: ", round(100 * max_proba.mean(), 1), "%") # 85.5%
print("Minimum of highest class probability is: ", round(100 * max_proba.min(), 1), "%") # 13.1%
print("Median of highest class probability is: ", round(100 * max_proba.median(), 1), "%") # 95.0%

THRESHOLD = [x / 100 for x in range(40, 75, 5)]
ACCURACIES = []
F1 = []
SAMPLE_FRACTION = []

for THRES in THRESHOLD:
    y_pred_thres = y_pred[max_proba > THRES]
    y_test_thres = y_test[max_proba > THRES]
    
    thres_acc = metrics.accuracy_score(y_test_thres, y_pred_thres)
    thres_f1 = metrics.f1_score(y_test_thres, y_pred_thres, average = "weighted")
    sample_fraction = len(y_test_thres) / len(y_test)
  
    ACCURACIES.append(thres_acc)
    F1.append(thres_f1)
    SAMPLE_FRACTION.append(sample_fraction)

eval_df = pd.DataFrame({"min_proba": ["No"] + THRESHOLD,
              "Accuracy": [acc] + ACCURACIES,
              "F1": [f1] + F1, 
              "Sample_Fraction": [1] + SAMPLE_FRACTION})
print(eval_df)
eval_df.to_csv(path + "/Data/training_evaluation_results/05a_max_proba_thresholds.csv")
print("Evaluation results for minimum probability thresholds saved.")

# group shares ----------------------
group_shares = pd.Series(y_test).value_counts()/len(y_test)
group_shares = group_shares.sort_index()
groups = pd.DataFrame(columns = ["ethnic origin", "baseline_share", "threshold", "N_samples", "sample_share"])

for THRES in THRESHOLD:
    N_samples = pd.Series(y_pred[max_proba > THRES])
    N_samples = N_samples.value_counts().sort_index()
    sample_share = [x / sum(N_samples) for x in N_samples]
    
    tmp = pd.DataFrame({"ethnic origin": N_samples.index, "baseline_share": list(group_shares),
                    "threshold": [THRES] * len(sample_share), "N_samples": list(N_samples), 
                    "sample_share": sample_share})
    groups = groups.append(tmp)
groups.to_csv(path + "/Data/training_evaluation_results/05b_max_proba_group_sizes.csv")
print("Group size evaluation for minimum probability thresholds saved.")

##########################################################################
###### Subsampling based on additional metrics to increase accuracy ######
##########################################################################

class_probas = xgb_model.predict_proba(x_test)

# difference to second highest origin probability
max_proba = [max(x) for x in class_probas]
pred_second = [sorted(x)[-2] for x in class_probas]
dist_second = pd.Series(max_proba)- pd.Series(pred_second)

# entropy among all origin probabilities
entro = [x * np.log(x) for x in class_probas]
entro = [-1 * np.nansum(x) for x in entro]

dat_eval = pd.DataFrame({"max_proba": max_proba, "dist_second": dist_second, "entro": entro})

# group shares
groups = pd.DataFrame(columns = ["ethnic origin", "baseline_share", "min_proba", 
                                "min_distance", "max_entropy", "N_samples", "sample_share"])
def group_size_eval(MAX_PROBA, DIST_SECOND, ENTRO):
    tmp = dat_eval[(dat_eval["max_proba"] >= MAX_PROBA) &
                   (dat_eval["entro"] <= ENTRO) &
                   (dat_eval["dist_second"] >= DIST_SECOND)]
    
    N_samples = pd.Series(y_pred[tmp.index])
    N_samples = N_samples.value_counts().sort_index()
    sample_share = [x / sum(N_samples) for x in N_samples]
    
    tmp = pd.DataFrame({"ethnic origin": N_samples.index, "baseline_share": list(group_shares),
                    "min_proba": MAX_PROBA, "min_distance": DIST_SECOND, "max_entropy": ENTRO,
                    "N_samples": list(N_samples), "sample_share": sample_share})
    return(tmp)

def acc_evaluate(MAX_PROBA, DIST_SECOND, ENTRO):
    tmp = dat_eval[(dat_eval["max_proba"] >= MAX_PROBA) &
                   (dat_eval["entro"] <= ENTRO) &
                   (dat_eval["dist_second"] >= DIST_SECOND)]
    
    y_pred_thres = y_pred[tmp.index]
    y_test_thres = y_test[tmp.index]
    
    thres_acc = metrics.accuracy_score(y_test_thres, y_pred_thres)
    thres_f1 = metrics.f1_score(y_test_thres, y_pred_thres, average = "weighted")
    sample_fraction = len(y_test_thres) / len(y_test)
    
    res_out = [MAX_PROBA, DIST_SECOND, ENTRO, 
               round(sample_fraction, 3), round(thres_acc, 3), round(thres_f1, 3)]
    return(res_out)

# define threshold values to evaluate
PROBAS = [0, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
DISTANCES = [0, 0.2, 0.3, 0.4, 0.5] 
ENTROPIES = [10, 2, 1.75]

res = pd.DataFrame(None, columns = ["min_proba", "min_distance",
                                    "max_entropy", "sample_fraction",
                                    "accuracy", "f1"])

# check accuracies for threshold combinations
for PROBA in PROBAS:
    for DIST in DISTANCES:
        for ENTROP in ENTROPIES:
            res_out = acc_evaluate(MAX_PROBA=PROBA, DIST_SECOND=DIST, ENTRO = ENTROP)
            res.loc[len(res)] = res_out
            groups_out = group_size_eval(MAX_PROBA=PROBA, DIST_SECOND=DIST, ENTRO = ENTROP)
            groups = groups.append(groups_out)
            
res = res.sort_values("f1", ascending = False)
res.to_csv(path + "/Data/training_evaluation_results/05c_indicator_thresholds.csv")
print("Evaluation results for indicator thresholds saved.")
groups.to_csv(path + "/Data/training_evaluation_results/05d_indicator_thresholds_group_sizes.csv")
print("Group size evaluation for indicator thresholds saved.")

##############################################
###### Use LSTM to check accuracies ##########
##############################################

import random
import tensorflow as tf
tf.random.set_seed(4032021)

# tokenize names
def encode_chars(names, seq_max, char_dict, n_chars):

    N = len(names)
    END_idx = np.where(pd.Series(char_dict) == "END")[0][0]
    
    # Create 3D-Tensor with shape (No. of samples, maximum name length, number of characters):
    tmp = np.zeros(shape = (N, seq_max, n_chars)) 

    # iterate over all names
    for i in range(N):
        name = names[i]
        
        # truncate at seq_max
        if(len(name) > seq_max):
            name = name[:seq_max]
        
        # encode characters
        for char in range(len(name)):
            idx_pos = np.where(pd.Series(char_dict) == name[char])[0][0]
            tmp[i, char, idx_pos] = 1
            
        # padding
        if len(name) < seq_max:
            tmp[i, len(name):seq_max, END_idx] = 1
    
    return tmp

CHAR_DICT = list([chr(i) for i in range(97,123)])+[" ", "END"]
SEQ_MAX = 30
N_CHARS = 28

# define a validation sample from RF testing sample
N_val = int(round(len(idx_test) * 0.2, 0))
np.random.seed(4032021)
val_idx = random.sample(set(idx_test), k = N_val)
NAMES = list(df.loc[val_idx, "Name"])

y_val = response[val_idx]
x_val = encode_chars(names = NAMES, char_dict = CHAR_DICT,
             seq_max = SEQ_MAX, n_chars = N_CHARS)
print('Testing Features Shape:', x_val.shape)
print('Testing Labels Shape:', y_val.shape)
print("Validation set defined.")

# define training samples & get their RF predictions
train_samples_idx = [i for i in idx_test if i not in val_idx]
train_samples_proba = xgb_model.predict_proba(features[train_samples_idx])

max_proba = [max(x) for x in train_samples_proba]
pred_second = [sorted(x)[-2] for x in train_samples_proba]
dist_second = pd.Series(max_proba)- pd.Series(pred_second)
entro = [x * np.log(x) for x in train_samples_proba]
entro = [-1 * np.nansum(x) for x in entro]

dat_eval = pd.DataFrame({"idx": train_samples_idx, "max_proba": max_proba, 
                         "dist_second": dist_second, "entro": entro})

# get training samples that fullfill a sepcific threshold combination
def subsample_train(MAX_PROBA, DIST_SECOND, ENTRO):
    tmp = dat_eval[(dat_eval["max_proba"] >= MAX_PROBA) &
                   (dat_eval["entro"] <= ENTRO) &
                   (dat_eval["dist_second"] >= DIST_SECOND)]
    return(tmp["idx"])

threshold_eval = pd.DataFrame(None, 
                              columns = list(dat_eval.columns)[1:] + ["training_sample_fraction", "accuracy"])
print("Training set of", len(dat_eval), "samples defined.")

# define thresholds
PROBAS = [0, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
DISTANCES = [0, 0.2, 0.3, 0.4, 0.5] 

# train model with different training samples according to thresholds and save accuracies
for PROBA in PROBAS:
    for DIST in DISTANCES:
        train_idx = subsample_train(MAX_PROBA = PROBA, DIST_SECOND = DIST, ENTRO = 2)
        NAMES = list(df.loc[train_idx, "Name"])
        SAMPLE_FRACTION = len(train_idx) / len(dat_eval)
        
        y_train = response[train_idx]
        x_train = encode_chars(names = NAMES, char_dict = CHAR_DICT,
                               seq_max = SEQ_MAX, n_chars = N_CHARS)
        
        # set up a simple LSTM
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(units = 128, return_sequences = True,
                                 input_shape = (x_train.shape[1], x_train.shape[2])),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.LSTM(units = 64, return_sequences = False),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(len(np.unique(y_test)), activation = "softmax")
            ])
        
        model.compile(optimizer =  "adam", 
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"])
        
        CALLBACK = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                            patience = 2, 
                                            restore_best_weights = True)
        
        hist = model.fit(x = x_train, y = y_train,
                         epochs = 7, batch_size = 64,
                         callbacks= [CALLBACK], 
                         verbose = 0,
                         validation_data = (x_val, y_val))
        
        ACC = round(model.evaluate(x_val, y_val)[1], 3)
        threshold_eval.loc[len(threshold_eval)] = [PROBA, DIST, 2, SAMPLE_FRACTION, ACC]
        
print("Evaluation based on LSTM model performance completed. Performance: ")
print(threshold_eval)
threshold_eval.to_csv(path + "/Data/training_evaluation_results/06_LSTM_performances.csv")



