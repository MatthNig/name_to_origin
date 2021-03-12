#################################################################
# Description:    Script to train the LSTM model to classify    #
#                 inventor origins based on names.              #
# Authors:        Matthias Niggli/CIEB UniBasel                 #
# Date:           21.10.2020                                    #
#################################################################

#### Import packages ---------------------------------------------------------
import tensorflow as tf
import numpy as np
import pandas as pd
import random
# import pyreadr
import os
import sys
# from matplotlib import pyplot as plt
print("All packages loaded.")

#### set seed for reproducibility --------------------------------------------
np.random.seed(10082020)
tf.random.set_seed(10082020)

#### Set directory-------------------------------------------------------------
path = "C:/Users/Matthias/Documents/GithubRepos/inventor_migration"
#path = "/scicore/home/weder/nigmat01/inventor_migration"
os.chdir(path)
os.getcwd()
print("Directories specified")

#### Load the data ------------------------------------------------------------
df_train = pd.read_csv("Data/training_data/df_train.csv")
print("Data for training the model successfully loaded.")

############################################
#### Encode data for training the model ####
############################################
function_path = path+"/Code/classification_models"
if function_path not in sys.path:
    sys.path.append(function_path)
from names_encoding_function import encode_chars

## Specify parameters for encoding:
CHAR_DICT = list([chr(i) for i in range(97,123)])+[" ", "END"]
SEQ_MAX = 30
N_CHARS = 28
NAMES = df_train["full_name"]

## encode names
x_dat = encode_chars(names = NAMES, char_dict = CHAR_DICT,
             seq_max = SEQ_MAX, n_chars = N_CHARS)
print("All names encoded")

## encode ethnical origin
y_classes = {"regions": sorted(list(df_train["origin"].unique())),
             "numbers": [i +1 for i in range(len(df_train["origin"].unique()))]}

y_classes = pd.DataFrame(y_classes)
y_dat = df_train["origin"].astype("category")
y_dat = y_dat.cat.rename_categories(list(y_classes["numbers"]))
y_dat = np.array(y_dat)
y_dat = tf.keras.utils.to_categorical(y_dat, dtype = "int64")
y_dat = y_dat[:, 1:]
print("All names classified and encoded for training the model.")
print("Data for training the model successfully transformed.")


#### Define train and test set ------------------------------------------------
train_frac = 0.8 # fraction of data to train the model
N = len(df_train)
train_idx = random.sample(range(N), k = int(round(N * train_frac, 0)))
val_idx = [i for i in range(N) if i not in train_idx]

x_train = x_dat[train_idx, :, :]
x_val = x_dat[val_idx, :, :]
y_train = y_dat[train_idx, :]
y_val = y_dat[val_idx, :]

############################################
#### Train the network #####################
############################################

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units = 512, return_sequences = True,
                         input_shape = (x_train.shape[1], x_train.shape[2])),
    tf.keras.layers.Dropout(0.33),
    tf.keras.layers.LSTM(units = 256, return_sequences = True),
    tf.keras.layers.Dropout(0.33),
    tf.keras.layers.LSTM(units = 64, return_sequences = False),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(y_dat.shape[1], activation = "softmax")
    ])
model.summary()

model.compile(optimizer =  "adam", 
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])

## class_weights: ------------------------------------------------------------
y_classes["class_weights"] = 1
y_classes.loc[y_classes.regions == "AngloSaxon", "class_weights"] = 10
CLASS_WEIGHTS = list(y_classes["class_weights"])
CLASS_WEIGHTS = dict(zip(y_classes["numbers"]-1, CLASS_WEIGHTS))

## training parameters: -------------------------------------------------------
EPOCHS = 20
BATCH_SIZE = 256
CALLBACK = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                            patience=3, 
                                            restore_best_weights = True)

hist = model.fit(x = x_train, y = y_train,
                 epochs = EPOCHS, batch_size = BATCH_SIZE,
                 class_weight = CLASS_WEIGHTS, 
                 callbacks= CALLBACK, verbose = 2,
                 validation_data= (x_val, y_val))
