#########################################################################
# Description:    Script to assign NamePrism Leaf Nationalities to      #
#                 ethnic origins.                                       #
# Authors:        Matthias Niggli/CIEB UniBasel                         #
# Last Revised:   02.03.2021                                            #
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

import joblib

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

# create a reference dictionary that maps ethnic origins to NamePrism Leaf Nationalities
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

#### (1) retrieve highest leaf nationality predicition and assign to corresponding ethnic origin   
pred_response_baseline = pd.DataFrame(x_test).idxmax(axis = 1)
pred_response_baseline = pd.DataFrame(pred_response_baseline, columns=["feature_idx"])
pred_response_baseline = pd.merge(pred_response_baseline, origin_dict, how = "left", 
                                  on = ["feature_idx"])
pred_response_baseline = pred_response_baseline.rename(columns={"origin_encoded": "predicted_origin_encoded", 
                                                                "origin": "predicted_origin"}) 
pred_response_baseline = pred_response_baseline.loc[:, ['predicted_origin_encoded', 'predicted_origin']]
pred_response_baseline["true_origin"] = y_test
pred_response_baseline = pred_response_baseline.dropna() # drop "Other" categories
acc = metrics.accuracy_score(pred_response_baseline["true_origin"], pred_response_baseline['predicted_origin_encoded'])
print("Overall accuracy when classifying to the highest leaf nationality is ", 
      round(acc * 100, 1), "%") # 69%

#### (2) map leaf nationalities to origins, sum them up and classify to highest probability
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
# acc = len(pred_response_baseline[(pred_response_baseline.match == 1)]) / len(pred_response_baseline)
print("Overall accuracy when classifying to the highest aggregate origin group is ", 
      round(acc * 100, 1), "%") # 66.3%

#### (3) add thresholds based on distance to second highest prediction and entropy
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
      round(100 * (1 - len(sub_res) / len(res)), 1), "% of samples.") # 38%
print("Overall accuracy imposing thresholds is", 
      round(acc_threshold * 100, 1), "%") # 83%

#################################
###### Train RandomForest #######
#################################

# (1) specify the model parameters and fit a baseline random forest
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
      round((rf_base_acc - acc) * 100, 1), "percentage points") # 18.4pp

#################################
##### Hyperparamater Tuning #####
#################################

# define random grid
N_TREE = [x for x in range(200, 2200, 200)]
MAX_FEATURES = ['auto', 'sqrt', 8, 12]
MIN_SPLIT = [2, 5, 10]
MIN_LEAF = [1, 2, 5, 10]
RF_GRID = {"n_estimators": N_TREE, "max_features": MAX_FEATURES,
           "min_samples_split": MIN_SPLIT, "min_samples_leaf": MIN_LEAF}

# find best parameters based on random grid search
rf = RandomForestClassifier() 
rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = RF_GRID, 
                               n_iter = 50, cv = 3, verbose = 2, 
                               random_state = 28022021)

# # use subsample for testing the code
# x_train, x_test, y_train, y_test = train_test_split(features, response, 
#                                                     test_size = 250, train_size = 750,
#                                                     random_state = 2032021)

rf_random = rf_random.fit(X = x_train, y = y_train)
rf_random.best_params_
rf_best_random = rf_random.best_estimator_

# compare accuracy to base random forest model
y_pred = rf_best_random.predict(x_test)
rf_random_acc = metrics.accuracy_score(y_test, y_pred)
print("Overall accuracy of the tuned Random Forest classifier is ", 
      round(rf_random_acc * 100, 1), "%") #xx.x%
print("Improvment against baseline random forest is ", 
      round((rf_random_acc - rf_base_acc) * 100, 1), "percentage points") # xx.x pp

#####################################################
###### Try sample weights to imporve learning #######
#####################################################

# use weights on samples that have been wrongly classified in baseline
y_pred = rf_best_random.predict(x_train) # predict on training sample
train_acc = metrics.accuracy_score(y_train, y_pred)
print("Accuracy on the training sample is ",
      round(train_acc * 100, 1), "%") # 95.1%

WEIGHTS = np.where(y_pred == y_train, 1, 1-train_acc)
rf_best_random_weights = rf_best_random.fit(X = x_train, y = y_train, sample_weight = WEIGHTS)

y_pred = rf_best_random_weights.predict(x_test) # predict on test sample
rf_random_weight_acc = metrics.accuracy_score(y_test, y_pred)
print("Overall accuracy of the Random Forest classifier with weights is ", 
      round(rf_random_weight_acc * 100, 1), "%") # 84.4%
print("Improvment against not using weights is ", 
      round((rf_random_weight_acc - rf_random_acc) * 100, 1), "percentage points") # 18.4pp

##################################
###### Save the best model #######
##################################

if rf_random_weight_acc > rf_random_acc:
    rf_save = rf_best_random_weights
    print("Save random forest model with sample weights.")
    WEIGHTS = "weights"
else: 
    rf_save = rf_best_random
    print("Save random forest model without sample weights.")
    WEIGHTS = "no_weights"

joblib.dump(rf_save, path + "/Classification_models/rf_origin_assignment_" + WEIGHTS + "_compressed.joblib", compress = 5)

print("Best performing model saved.")
