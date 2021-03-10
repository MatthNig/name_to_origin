#########################################################################
# Description:    Script to assign NamePrism Leaf Nationalities to      #
#                 ethnic origins based on manual crosswalkd.            #
# Authors:        Matthias Niggli/CIEB UniBasel                         #
# Last Revised:   10.03.2021                                            #
#########################################################################

#######################################
###### Load packages and data #########
#######################################

#### Import packages ---------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
print("All packages loaded.")

#### Set directory ---------------------------------------------------------
path = "C:/Users/Matthias/Documents/GithubRepos/name_to_origin"
#path = "/scicore/home/weder/nigmat01/name_to_origin"
print("Directories specified")

#### Load & process the data -------------------------------------------------
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
pred_baseline = pd.DataFrame(x_test).idxmax(axis = 1)
pred_baseline = pd.DataFrame(pred_baseline, columns=["feature_idx"])
pred_baseline = pd.merge(pred_baseline, origin_dict, how = "left", 
                                  on = ["feature_idx"])
pred_baseline = pred_baseline.rename(columns={"origin_encoded": "predicted_origin_encoded", 
                                                                "origin": "predicted_origin"}) 
pred_baseline = pred_baseline.loc[:, ['predicted_origin_encoded', 'predicted_origin']]
pred_baseline["true_origin"] = y_test
pred_baseline = pred_baseline.dropna() # drop NA categories
acc = metrics.accuracy_score(pred_baseline["true_origin"], pred_baseline['predicted_origin_encoded'])
print("Overall accuracy when classifying to the highest leaf nationality is ", 
      round(acc * 100, 1), "%") # 69%
f1 = metrics.f1_score(y_true = pred_baseline["true_origin"],
                      y_pred = pred_baseline['predicted_origin_encoded'],
                      average = "weighted")
print("Weighted F1 score when classifying to the highest leaf nationality is ", 
      round(f1 * 100, 1), "%") # 67.7%

# (3) aggregate leaf nationality probabilities per ethnic origin and classify to highest origin probability
df_cols = ["true_origin"] + list(range(17))
pred_baseline = pd.DataFrame(columns = df_cols)
pred_baseline["true_origin"] = y_test
for i in range(len(df_cols)-1):
    CODE = pred_baseline.columns[i+1]
    LEAF_NAT = origin_dict[origin_dict.origin_encoded == CODE]["feature_idx"]
    ORIGIN_PROB = x_test[:, LEAF_NAT].sum(axis = 1)
    pred_baseline.iloc[:, i+1] = ORIGIN_PROB
pred_baseline["max_pred"] = pred_baseline.iloc[:,1:].max(axis = 1)
pred_baseline["predicted_origin_encoded"] = pred_baseline.iloc[:,1:].idxmax(axis = 1)
pred_baseline["match"] = np.where(
    pred_baseline['predicted_origin_encoded'] == pred_baseline["true_origin"], 1, 0)
acc = metrics.accuracy_score(pred_baseline["true_origin"], pred_baseline['predicted_origin_encoded'])
print("Overall accuracy when classifying to the highest aggregate origin group is ", 
      round(acc * 100, 1), "%") # 66.3%
f1 = metrics.f1_score(y_true = pred_baseline["true_origin"],
                      y_pred = pred_baseline['predicted_origin_encoded'],
                      average = "weighted")
print("Weighted F1 score when classifying to the highest aggregate origin group is ", 
      round(f1 * 100, 1), "%") # 65.3%

# (4) add thresholds based on minimum prediction probability, 
#     distance to second highest prediction and entropy.
pred_second = []
drop_cols = pred_baseline.iloc[:, 1:(len(pred_baseline.columns)-3)].idxmax(axis = 1)
for i in range(len(pred_baseline)):
    cols = [x + 1 for x in list(pred_baseline.columns) if 
            x not in [drop_cols[i], "true_origin", "max_pred", "predicted_origin_encoded", "match"]]
    pred_second.append(pred_baseline.iloc[i,cols].max())
pred_baseline["pred_second"] = pred_second
pred_baseline["dist_second"] = pred_baseline["max_pred"] - pred_second

entro = pred_baseline.iloc[:, 1:18] 
entro = entro * np.log(entro)
entro = - entro.sum(axis = 1)
pred_baseline["pred_entropy"] = entro

THRES_MAX_PRED = 0.6
THRES_DIST_SECOND = 0.2
THRES_ENTROPY = 2

res = pred_baseline
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
