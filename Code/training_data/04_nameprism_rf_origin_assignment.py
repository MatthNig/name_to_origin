#########################################################################
# Description:    Script to assign NamePrism Leaf Nationalities to      #
#                 ethnic origins.                                       #
# Authors:        Matthias Niggli/CIEB UniBasel                         #
# Last Revised:   08.03.2021                                            #
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
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb

import tensorflow as tf

#import joblib

print("All packages loaded.")

#### Set directory ---------------------------------------------------------
#path = "C:/Users/Matthias/Documents/GithubRepos/name_to_origin"
path = "/scicore/home/weder/nigmat01/name_to_origin"
print("Directories specified")

#### Load & process the data -------------------------------------------------
df = pd.read_csv(path+"/Data/nameprism_stratified_sample.csv")
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

x_train, x_test, y_train, y_test = train_test_split(features, response, 
                                                    test_size = 0.2, 
                                                    random_state = 25022021)
print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)


#############################################################
###### Define Baseline Classification performances ##########
#############################################################

# (1) create a manual crosswalk that maps ethnic origins to NamePrism Leaf Nationalities
origin_dict = df[["origin_encoded", "origin"]].drop_duplicates()
leaf_nationalities = pd.DataFrame(df.columns[1:-1], columns = ["leaf_nationality"])
leaf_nationalities["origin"] = ["Balkans", "Arabic", "Italian", "EastEurope",
                              "Other", "Other", "Other", "French",
                              "SouthEastAsia", "SouthEastAsia", "Other", "Other",
                              "SouthEastAsia", "Scandinavian", "SouthEastAsia", "Scandinavian",
                              "Persian", "Scandinavian", "Arabic", "Other", 
                              "Arabic", "Hispanic-Iberian", "Slavic-Russian", "Arabic",
                              "Other", "Japan", "German", "China",
                              "India", "Hispanic-Iberian", "Scandinavian", "Turkey",
                              "Other", "AngloSaxon", "SouthEastAsia", "Korea",
                              "Other", "EastEurope", "SouthEastAsia"]
leaf_nationalities["feature_idx"] = pd.Series(range(39))
origin_dict = pd.merge(origin_dict, leaf_nationalities, how = "left", on = ["origin"])

# (2) retrieve highest leaf nationality predicition and assign to corresponding ethnic origin   
pred_response_baseline = pd.DataFrame(x_test).idxmax(axis = 1)
pred_response_baseline = pd.DataFrame(pred_response_baseline, columns=["feature_idx"])
pred_response_baseline = pd.merge(pred_response_baseline, origin_dict, how = "left", 
                                  on = ["feature_idx"])
pred_response_baseline = pred_response_baseline.rename(columns={"origin_encoded": "predicted_origin_encoded", 
                                                                "origin": "predicted_origin"}) 
pred_response_baseline = pred_response_baseline.loc[:, ['predicted_origin_encoded', 'predicted_origin']]
pred_response_baseline["true_origin"] = y_test
pred_response_baseline = pred_response_baseline.dropna() # drop NA categories
acc = metrics.accuracy_score(pred_response_baseline["true_origin"], pred_response_baseline['predicted_origin_encoded'])
print("Overall accuracy when classifying to the highest leaf nationality is ", 
      round(acc * 100, 1), "%") # 69%
f1 = metrics.f1_score(y_true = pred_response_baseline["true_origin"],
                      y_pred = pred_response_baseline['predicted_origin_encoded'],
                      average = "weighted")
print("Weighted F1 score when classifying to the highest leaf nationality is ", 
      round(f1 * 100, 1), "%") # 67.7%

# (3) aggregate leaf nationality probabilities per ethnic origin and classify to highest origin probability
df_cols = ["true_origin"] + list(range(17))
pred_response_baseline = pd.DataFrame(columns = df_cols)
pred_response_baseline["true_origin"] = y_test
for i in range(len(df_cols)-1):
    CODE = pred_response_baseline.columns[i+1]
    LEAF_NAT = origin_dict[origin_dict.origin_encoded == CODE]["feature_idx"]
    ORIGIN_PROB = x_test[:, LEAF_NAT].sum(axis = 1)
    pred_response_baseline.iloc[:, i+1] = ORIGIN_PROB
pred_response_baseline["max_pred"] = pred_response_baseline.iloc[:,1:].max(axis = 1)
pred_response_baseline["predicted_origin_encoded"] = pred_response_baseline.iloc[:,1:].idxmax(axis = 1)
pred_response_baseline["match"] = np.where(
    pred_response_baseline['predicted_origin_encoded'] == pred_response_baseline["true_origin"], 1, 0)
acc = metrics.accuracy_score(pred_response_baseline["true_origin"], pred_response_baseline['predicted_origin_encoded'])
print("Overall accuracy when classifying to the highest aggregate origin group is ", 
      round(acc * 100, 1), "%") # 66.3%
f1 = metrics.f1_score(y_true = pred_response_baseline["true_origin"],
                      y_pred = pred_response_baseline['predicted_origin_encoded'],
                      average = "weighted")
print("Weighted F1 score when classifying to the highest aggregate origin group is ", 
      round(f1 * 100, 1), "%") # 65.3%

# (4) add thresholds based on minimum prediction probability, 
#     distance to second highest prediction and entropy.
pred_second = []
drop_cols = pred_response_baseline.iloc[:, 1:(len(pred_response_baseline.columns)-3)].idxmax(axis = 1)
for i in range(len(pred_response_baseline)):
    cols = [x + 1 for x in list(pred_response_baseline.columns) if 
            x not in [drop_cols[i], "true_origin", "max_pred", "predicted_origin_encoded", "match"]]
    pred_second.append(pred_response_baseline.iloc[i,cols].max())
pred_response_baseline["pred_second"] = pred_second
pred_response_baseline["dist_second"] = pred_response_baseline["max_pred"] - pred_second

entro = pred_response_baseline.iloc[:, 1:18] 
entro = entro * np.log(entro)
entro = - entro.sum(axis = 1)
pred_response_baseline["pred_entropy"] = entro

THRES_MAX_PRED = 0.6
THRES_DIST_SECOND = 0.2
THRES_ENTROPY = 2

res = pred_response_baseline
sub_res = res.iloc[np.where(res.max_pred >= THRES_MAX_PRED)]
sub_res = sub_res.iloc[np.where(sub_res.dist_second >= THRES_DIST_SECOND)]
sub_res = sub_res.iloc[np.where(sub_res.pred_entropy <= THRES_ENTROPY)]
acc_threshold = metrics.accuracy_score(sub_res["true_origin"], sub_res['predicted_origin_encoded'])
print("Imposing thresholds drops ", 
      round(100 * (1 - len(sub_res) / len(res)), 1), "% of samples.") # 38.3%
print("Overall accuracy imposing thresholds is", 
      round(acc_threshold * 100, 1), "%") # 83.0%
f1 = metrics.f1_score(y_true = sub_res["true_origin"],
                      y_pred = sub_res['predicted_origin_encoded'],
                      average = "weighted")
print("Weighted F1 score imposing thresholds is", 
      round(f1 * 100, 1), "%") # 81.6%

#################################
###### Train RandomForest #######
#################################

# (1) specify the model parameters and fit a baseline random forest ------------------
N_TREE = 200
rf = RandomForestClassifier(n_estimators = N_TREE, max_features = "sqrt",
                            min_samples_split = 5, min_samples_leaf = 3,
                            random_state= 28022021)
rf = rf.fit(X = x_train, y = y_train)
y_pred = rf.predict(x_test)
rf_base_acc = metrics.accuracy_score(y_test, y_pred)
print("Overall accuracy of the Random Forest classifier is ", 
      round(rf_base_acc * 100, 1), "%") # 84.6%
print("Improvment against not using a model is ", 
      round((rf_base_acc - acc) * 100, 1), "percentage points") # 18.3pp
f1 = metrics.f1_score(y_true = y_test, y_pred = y_pred, average = "weighted")
print("Weighted F1 score of Random Forest Classifier is: ", 
      round(f1 * 100, 1), "%") # 84.5%

# (2) Hyperparamater Tuning Random Forest --------------------------------------
N_TREE = [x for x in range(200, 2200, 200)]
MAX_FEATURES = ['auto', 'sqrt', 8, 12]
MIN_SPLIT = [2, 5, 8]
MIN_LEAF = [1, 2, 5]
RF_GRID = {"n_estimators": N_TREE, "max_features": MAX_FEATURES,
           "min_samples_split": MIN_SPLIT, "min_samples_leaf": MIN_LEAF}
print("Searching through", 
      len(N_TREE) * len(MAX_FEATURES) * len(MIN_SPLIT) * len(MIN_LEAF),
      "parameter combinations.")


# find best parameters based on random grid search
rf = RandomForestClassifier() 
rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = RF_GRID, 
                               n_iter = 50, cv = 3, verbose = 0, 
                               random_state = 28022021)

""" # use subsample for testing the code
x_train, x_test, y_train, y_test = train_test_split(features, response, 
                                                    test_size = 250, train_size = 750,
                                                    random_state = 2032021)
 """
rf_random = rf_random.fit(X = x_train, y = y_train)
print("Parameters of the tuned random forest: ")
print(rf_random.best_params_) # 'max_features': 'auto', 'min_samples_leaf': 1, min_samples_split': 2, 'n_estimators': 1400
rf_best_random = rf_random.best_estimator_

# compare accuracy to base random forest model
y_pred = rf_best_random.predict(x_test)
rf_random_acc = metrics.accuracy_score(y_test, y_pred)
print("Overall accuracy of the tuned Random Forest classifier is ", 
      round(rf_random_acc * 100, 1), "%") #85.0%
print("Improvment against baseline random forest is ", 
      round((rf_random_acc - rf_base_acc) * 100, 1), "percentage points") # 0.4pp
f1 = metrics.f1_score(y_true = y_test, y_pred = y_pred, average = "weighted")
print("Weighted F1 score of tuned Random Forest Classifier is: ", 
      round(f1 * 100, 1), "%") # xx.x%

# (3) Sample weights to imporve learning of Random Forest -------------------------------------
# use weights on samples that have been wrongly classified in baseline
y_pred = rf_best_random.predict(x_train) # predict on training sample
train_acc = metrics.accuracy_score(y_train, y_pred)
print("Accuracy on the training sample is ",
      round(train_acc * 100, 1), "%") # 99.1%

WEIGHTS = np.where(y_pred == y_train, 1, 1-train_acc)
rf_best_random_weights = rf_best_random.fit(X = x_train, y = y_train, sample_weight = WEIGHTS)

y_pred = rf_best_random_weights.predict(x_test) # predict on test sample
rf_random_weight_acc = metrics.accuracy_score(y_test, y_pred)
print("Overall accuracy of the Random Forest classifier with weights is ", 
      round(rf_random_weight_acc * 100, 1), "%") # 85.0%
print("Improvment against not using weights is ", 
      round((rf_random_weight_acc - rf_random_acc) * 100, 1), "percentage points") # 0.0 pp

""" # (4) Save the best Random Forest model --------------------------------------------------------

if rf_random_weight_acc > rf_random_acc:
    rf_save = rf_best_random_weights
    print("Save random forest model with sample weights.")
    WEIGHTS = "weights"
else: 
    rf_save = rf_best_random
    print("Save random forest model without sample weights.")
    WEIGHTS = "no_weights"

joblib.dump(rf_save, path + "/Classification_models/rf_origin_assignment_" + WEIGHTS + "_compressed.joblib", compress = 5)

print("Best performing random forest model saved.")
 """
############################
###### Train XGBoots #######
############################

# (1) Baseline --------------------------------------------------------
xgb_model = xgb.XGBClassifier(random_state = 8032021, n_estimators = 100, learning_rate = 0.1)
xgb_model.fit(X = x_train, y = y_train)
y_pred = xgb_model.predict(x_test)
xgb_base_acc = metrics.accuracy_score(y_test, y_pred)
print("Overall accuracy of the XGBoost is ", 
      round(xgb_base_acc * 100, 1), "%") # 85.2%
f1 = metrics.f1_score(y_true = y_test, y_pred = y_pred, average = "weighted")
print("Weighted F1 score of XGBoost is: ", 
      round(f1 * 100, 1), "%") # 85.1%

# (2) Hyperparameter Tuning ------------------------------------------
N_ESTIMATORS = [x for x in range(60, 240, 40)]
MAX_DEPTH = [x for x in range(1, 11)]
LEARN_RATE = [0.1, 0.05, 0.01]
XG_GRID = {"n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH,
           "learning_rate": LEARN_RATE}
print("Searching through", 
      len(N_ESTIMATORS) * len(MAX_DEPTH) * len(LEARN_RATE),
      "parameter combinations.")

""" # use subsample for testing the code
x_train, x_test, y_train, y_test = train_test_split(features, response, 
                                                    test_size = 250, train_size = 750,
                                                    random_state = 8032021)
 """
xgb_model = xgb.XGBClassifier()
xgb_random = RandomizedSearchCV(estimator = xgb_model, 
                               param_distributions = XG_GRID, 
                               n_iter = 50, cv = 3, verbose = 0, 
                               random_state = 8032021)
xgb_random = xgb_random.fit(X = x_train, y = y_train)
xgb_best_random = xgb_random.best_estimator_
y_pred = xgb_best_random.predict(x_test)
xgb_random_acc = metrics.accuracy_score(y_test, y_pred)
print("Overall accuracy of the tuned XGBoost is ", 
      round(xgb_random_acc * 100, 1), "%") # xx.x%
f1 = metrics.f1_score(y_true = y_test, y_pred = y_pred, average = "weighted")
print("Weighted F1 score of XGBoost is: ", 
      round(f1 * 100, 1), "%") # xx.x%
print("Best parameters of the tuned XGBoost are:")
print(xgb_random.best_params_)

######################################
###### Train FF Neural Network #######
######################################

tf.random.set_seed(8032021)
#tf.random.set_random_seed(8032021) # TF1 on the cluster

res = pd.DataFrame(columns=["EPOCHS", "BATCH_SIZE", "NODES", "ACTIV", "acc", "f1"])

def FFNN_fit(N_EPOCH, BATCH_SIZE, N_NODE, ACTIV):
      FFNN_model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape = x_train.shape[1]),
            tf.keras.layers.Dense(units = N_NODE, activation = ACTIV),
            tf.keras.layers.Dense(len(np.unique(y_test)), activation = "softmax")
            ])
            
      FFNN_model.compile(optimizer =  "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
      
      CALLBACK = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 3, restore_best_weights = True)
      
      hist = FFNN_model.fit(x = x_train, y = y_train,
                 epochs = N_EPOCH, batch_size = BATCH_SIZE,
                 callbacks = [CALLBACK], verbose = 0,
                 validation_data= (x_test, y_test))
      
      y_pred = np.argmax(FFNN_model.predict(x_test), axis=-1)
      acc = metrics.accuracy_score(y_test, y_pred)
      f1 = metrics.f1_score(y_true = y_test, y_pred = y_pred, average = "weighted")

      tmp = [N_EPOCH, BATCH_SIZE, N_NODE, ACTIV, acc, f1]

      return tmp

# set the parameter values
EPOCHS = [20, 30]
BATCHES = [64, 128, 256]
NODES = [512, 256, 128, 64]
ACTIVS = ["relu", "softmax"]

# train different models and store the results:
for EPOCH in EPOCHS:
      for BATCH in BATCHES:
            for NODE in NODES:
                  for ACTIVATION in ACTIVS:
                        tmp = FFNN_fit(N_EPOCH = EPOCH, BATCH_SIZE = BATCH,
                                    N_NODE = NODE, ACTIV = ACTIVATION)
                        res.loc[len(res), :] = tmp

FFNN_acc = res["acc"].max()
idx = np.argmax(res["acc"])
print("Best network parameters: \n", res.iloc[idx, :4])
print("Overall accuracy of the best Feed-Forward Neural Network is ", 
      round(FFNN_acc * 100, 1), "%") # xx.x%
print("Weighted F1 score of Feed-Forward Neural Network is: ", 
      round(f1 * 100, 1), "%") # xx.x%




