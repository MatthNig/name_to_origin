### final sample:

#######################################
## Load packages and set directories ##
#######################################

# packages for data processing: ------------------------------------------------
library("tidyverse")

# packages for training the network: -------------------------------------------
library("tensorflow")
library("keras")
library("reticulate")

# load encoding function and function parameters--------------------------------
source(file = paste0(getwd(), "/Code/model_training/names_encoding_function.R"))
PARAMS <- read.csv(file = paste0(getwd(), "/Data/PARAMS.csv"))
SEQ_MAX <- PARAMS$SEQ_MAX
N_CHARS <- PARAMS$N_CHARS
CHAR_DICT <- read.csv(file = paste0(getwd(), "/Data/CHAR_DICT.csv"))
CHAR_DICT <- CHAR_DICT$x
PARAMS <- NULL
print("Function and function parameters loaded.")

# directories
if(substr(x = getwd(), 
          nchar(getwd())-13, nchar(getwd())) == "name_to_origin"){
  print("Working directory corresponds to repository directory")}else{
    print("Make sure your working directory is the repository directory.")}

# reproducibility
set.seed(26012021)

#########################
## Load & process data ##
#########################

# create training data
athlete_sample <- read.csv("Data/athlete_sample.csv") %>% rename(full_name = Name) %>% select(-Year)
inventor_sample <- read.csv("Data/inventor_sample.csv")
df_train <- rbind(athlete_sample, inventor_sample)

# subset to classes
df_train$origin <- ifelse(df_train$origin == "Portugese", "Hispanic", df_train$origin)
df_train$origin <- ifelse(df_train$origin == "Slavic", "Slawic", df_train$origin)
df_train <- filter(df_train, !origin %in% c("Africa", "Philippines", "Greek", "Jewish"))

# summarize
df_train %>% group_by(origin) %>% 
  summarize(count = n(),
            share = count / nrow(df_train)) %>%
  arrange(-count)

# shuffle the data
df_train <- df_train[sample(nrow(df_train), nrow(df_train)), ] # shuffle the data
rownames(df_train) <- NULL
inventor_sample <- NULL
athlete_sample <- NULL

#####################################################
######### encode the features and outcomes ##########
#####################################################

#### outcome: origin classes
y_classes <- data.frame(
  levels = levels(as.factor(df_train$origin)),
  numbers = seq(length(unique(df_train$origin)))
)
y_dat <- as.factor(df_train$origin)
levels(y_dat) <- y_classes$numbers
y_dat <- as.numeric(as.character(y_dat))
y_dat <- to_categorical(y_dat)
y_dat <- y_dat[, -1]
print("All names classified and encoded for training the model.")

#### features: names encoding
x_dat <- encode_chars(names = df_train$full_name,
                      seq_max = SEQ_MAX,
                      char_dict = CHAR_DICT,
                      n_chars = N_CHARS)
paste("names are one-hot-encoded with shape: ", 
      paste0("(", paste(dim(x_dat), collapse = ", "), ")")
)

################################################
######### split to train and test set ##########
################################################

train_frac <- 0.8 # fraction of data to train the model
N <- nrow(df_train)
train_idx <- sample(seq(N), N * train_frac)

x_train <- x_dat[train_idx, , ]
x_val <- x_dat[-train_idx, , ]
y_train <- y_dat[train_idx, ]
y_val <- y_dat[-train_idx, ]

#########################################
############ train the network ##########
#########################################

## build and compile the model -------------------------------------------------
model <- keras_model_sequential()
model %>%
  layer_lstm(units = 512, return_sequences = TRUE,
             input_shape = c(ncol(x_train), dim(x_train)[3])) %>%
  layer_dropout(rate = 0.33) %>%
  layer_lstm(units = 256, return_sequences = TRUE) %>%
  layer_dropout(rate = 0.33) %>%
  layer_lstm(units = 64) %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = ncol(y_train), activation = "softmax")
summary(model)

model %>% compile(
  optimizer =  "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy"))

## class_weights: --------------------------------------------------------------
# y_classes$class_weights <- ifelse(y_classes$levels  == "AngloSaxon", 10, 1)
# CLASS_WEIGHTS <- as.list(y_classes$class_weights)
# names(CLASS_WEIGHTS) <- y_classes$numbers

## training parameters--------------------------------------------------------
EPOCHS <- 20
BATCH_SIZE <- 512#256

## fit the model
hist <- model %>% fit(
  x = x_train, y = y_train, 
  # class_weights = CLASS_WEIGHTS,
  validation_data = list(x_val, y_val),
  callbacks = list(callback_early_stopping(monitor = "val_loss", 
                                           patience = 3, 
                                           restore_best_weights = TRUE)),
  epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 2)

## 26.01.2021
# (1)   NO CLASS WEIGHTS
#       TOTAL_ACCURACY: 85.0%, WEIGHTED_AVERAGE_F1: xx.x%, 
#       ANGLOSAXON_ACC: xx.x%, ANGLOSAXON_F1: 0.823, ANGLOSAXON_PRECISION:  0.792, ANGLOSAXON_RECALL:  0.858  
#       REAMRKS: weight initialization might help get better performance faster

#########################################
############ evaluate the model #########
#########################################

# for precision, recall and f1 in multiclass problems:
#https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1

## predict classes on validation set:
tmp <- df_train[-train_idx, ]
tmp$pred <- as.numeric(model %>% predict_classes(x_val[, ,]))
tmp$pred <- y_classes[tmp$pred + 1, "levels"]
tmp$res <- tmp$origin == tmp$pred

## overall accuracy of the model on evaluation set: -----------------------------------
res <- evaluate(model, x = x_val, y = y_val, verbose = 0)
table(tmp$res) / nrow(tmp)

# confusion matrix by origin -----------------------------------------------------------
conf_matrix_fun <- function(region){
  tmp <- tmp %>% filter(origin == region | pred == region)
  tmp <- tmp %>% mutate(origin = ifelse(origin == region, 1, 0),
                        pred = ifelse(pred == region, 1, 0))
  tmp <- table(tmp$origin, tmp$pred)
  return(tmp)
}
conf_matrix_fun("AngloSaxon")

## Precision by origin: --------------------------------------------------------
# i.e. "how many of the predicted origin are indeed of this origin?" (TP / TP + FP)
# "From all predicted AngloSaxons, how many are indeed AngloSaxons?"
origin_precision <- tmp %>% group_by(pred) %>% summarise(n_obs = n(), 
                                                         origin_precision = sum(res == TRUE) / n()) %>%
  rename(region = pred) %>%
  arrange(origin_precision)
origin_precision %>% arrange(origin_precision)
mean(origin_precision$origin_precision)

## Recall by origin: -----------------------------------------------------------
# "how many form a given origin are also predicted to be of this origin?" (TP / TP + FN)
# "how many AngloSaxons are indeed predicted as such?"
origin_recall <- tmp %>% group_by(origin) %>% summarise(n_obs = n(), 
                                                        origin_recall = sum(res == TRUE) / n()) %>%
  rename(region = origin) %>%
  arrange(origin_recall)
origin_recall %>% arrange(origin_recall)
mean(origin_recall$origin_recall)

## F1 by origin ----------------------------------------------------------------
origin_eval <- merge(origin_precision[, c("region", "origin_precision")],
                     origin_recall[, c("region", "origin_recall")], by = "region")
origin_eval$f1 <- 2 * (origin_eval$origin_precision * origin_eval$origin_recall) / 
  (origin_eval$origin_precision + origin_eval$origin_recall)
origin_eval %>% arrange(f1)

# weighted average in test data:
weights <- df_train %>% group_by(origin) %>% 
  summarize(N = n(),
            weight = N/nrow(df_train)) %>% 
  rename(region = origin)
origin_eval <- merge(origin_eval, weights, by = "region")
weighted.mean(x = origin_eval$f1, w = origin_eval$weight)

# Accuracy by origin: ------------------------------------------
acc_fun <- function(ctry){
  conf_matrix <- conf_matrix_fun(ctry)
  acc <- sum(diag(conf_matrix)) / sum(conf_matrix)
  return(acc)
}
origin_eval$accuracy <- unlist(lapply(origin_eval$region, acc_fun))
origin_eval %>% arrange(accuracy)

## median values and range of values across classes:
sapply(origin_eval[,-1], median)
sapply(origin_eval[,-1], mean)
sapply(origin_eval[,-1], range)

####################################
######### SAVE THE MODEL ###########
####################################

model %>% save_model_hdf5(file = paste0(getwd(), "/Classification_models/name_to_origin_LSTM.h5"))


## To-Do's----------------------------------------------
## 1) weight initialization
## 2) higher drop-out to prevent overfitting but be careful
## 3) larger batch size to ensure that there are a few labels of each class per batch
## 4) add early stopping & weight decay
## 5) maybe make a smaller model

## resources
## https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
## https://karpathy.github.io/2019/04/25/recipe/#2-set-up-the-end-to-end-trainingevaluation-skeleton--get-dumb-baselines
