#########################################################################
# Description:    Script to calculate optimal values for classification #
#                 of name-prism predictions to ethnic origins           #
# Authors:        Matthias Niggli/CIEB UniBasel                         #
# Date:           25.02.2021                                            #
#########################################################################

#######################################
###### Load packages and data #########
#######################################

# packages for data processing: ------------------------------------------------
library("tidyverse")

# Data -------------------------------------------------------------------------
df <- read.csv("Data/API_verification_sample.csv")

######################################################################
### Assign NamePrism Leaf Nationalities to Ethnic Origins ############
######################################################################

name_prism_origin <- names(df)[5:43]
ref_list <- data.frame(name_prism_origin = name_prism_origin,
                       origin_group = NA)
origins <- c("Balkans", "Arabic", "Italian", "EastEurope",
             "Africa", "Italian", "Africa","French", "SouthEastAsia",
             "SouthEastAsia", "Jewish", "CentralAsia", "SouthEastAsia",
             "Scandinavian", "SouthEastAsia", "Scandinavian", "Persian", 
             "Scandinavian", "Arabic", "Greek", 
             "Arabic", "Hispanic-Iberian", "Slavic-Russian", "Arabic", "Africa", "Japan",
             "German", "China", "India", "Hispanic-Iberian", "Scandinavian", "Turkey",
             "Philippines", "AngloSaxon", "SouthEastAsia", "Korea", "Africa",
             "EastEurope", "SouthEastAsia")
ref_list$origin_group <- origins
print("Assigned NamePrism Leaf Nationalities to Ethnic Origins")

###########################################################
######### Calculate Ethnic origin class probabilities #####
###########################################################

# prepare for classification
N = nrow(df)
n_X <- length(unique(ref_list$origin_group))+2
origin_groups <- as.data.frame(matrix(rep(NA, N * n_X), nrow = N))
colnames(origin_groups) <- c("Name", "Origin",
                             unique(ref_list$origin_group))
origin_groups[, c("Name", "Origin")] <- df[, c("Name", "origin")]

# for each name in the sample, calculate the ethnic origins probabilities 
# by summing up the corresponding NamePrism probabilities
for(i in 3:ncol(origin_groups)){
  origin_group <- colnames(origin_groups)[i]
  origins <- ref_list[ref_list$origin_group == origin_group, "name_prism_origin"]
  
  if(is.data.frame(df[, origins]) == FALSE){
    origin_prob <- df[, origins]}else{
      origin_prob <- rowSums(df[, origins])}
  
  origin_groups[, origin_group] <- origin_prob
}

df <- origin_groups
df$Origin <- ifelse(df$Origin == "Slawic", "Slavic-Russian", df$Origin)
df$Origin <- ifelse(df$Origin == "Hispanic", "Hispanic-Iberian", df$Origin)
paste0("Ethnic class probabilities calculated for ", nrow(df), " observations")

#############################################################
############## Construct additional indicators ##############
#############################################################

origins <- names(df)[3:ncol(df)]

# get maximum prediction
df$max_pred <- sapply(seq(1, nrow(df)), function(i){
  max_pred <- as.numeric(df[i, origins])
  max_pred <- max(max_pred, na.rm = TRUE)
  return(max_pred)
})

# calculate distance to second highest prediction
df$dist_2nd <- sapply(seq(1, nrow(df)), function(i){
  preds <- as.numeric(df[i, origins])
  preds <- sort(preds, decreasing = TRUE)
  dist_2nd <- preds[1] - preds[2]
  return(dist_2nd)
})

# calculate entropy
df$entropy <- sapply(seq(1, nrow(df)), function(i){
  entro <- as.numeric(df[i, origins])
  entro <- entro * log(entro)
  entro <- -sum(entro)
  return(entro)
  })

print("Calculated maximum class probability, distance to 2nd highest class probability and entropy.")

###########################################################################
####### Indicate how well NamePrism identifys Ethnic Origins ##############
###########################################################################

df$pred_origin <- sapply(seq(1, nrow(df)), function(i){
  max_pred_idx <- which(names(df) == "max_pred")
  pred_origins <- which(df[i, -max_pred_idx] == df$max_pred[i])
  pred_origins <- names(df)[pred_origins]
  return(pred_origins)
})

df$match <- ifelse(df$pred_origin == df$Origin, 1, 0)

# unconditional:
tmp <- table(df$match) / nrow(df)
paste("Overall Accuracy not imposing any thresholds is", 100 * round(tmp[2], 3), "%")

# conditional
tmp <- df %>% filter(max_pred >= 0.6 & dist_2nd >= 0.2 & entropy <= 2)
tmp <- table(tmp$match) / nrow(tmp)
paste("Overall Accuracy imposing  thresholds is", 100 * round(tmp[2], 3), "%")

if(100 * round(tmp[2], 3) <= 0.75){warning("Accuracy is not sufficent")}else{print("Accuracy is sufficient.")}

###########################################################
##### Save data for training a classification model #######
###########################################################

write.csv(df, file = "Data/NamePrism_class_threshold_trainsample.csv", row.names = FALSE)

