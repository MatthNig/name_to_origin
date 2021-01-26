#########################################################################
# Description:    Script to calculate optimal values for classification #
#                 of name-prism predictions to ethnic origins           #
# Authors:        Matthias Niggli/CIEB UniBasel                         #
# Date:           24.01.2021                                            #
#########################################################################

#######################################
## Load packages and set directories ##
#######################################

# packages for data processing: ------------------------------------------------
library("tidyverse")

# directories and reproducibility
datDir <- "/scicore/home/weder/nigmat01/Data"
setwd(datDir)

#######################################
############## Load data ##############
#######################################

df <- read.csv("API_verification_sample.csv")

###############################################
############ Create ethnic origins ############
###############################################

name_prism_origin <- names(df)[4:42]
ref_list <- data.frame(name_prism_origin = name_prism_origin,
                       origin_group = NA)
ref_list[grep("Africa", ref_list$name_prism_origin), 
         "origin_group"] <- "Africa"
ref_list[grep("Europe", ref_list$name_prism_origin), 
         "origin_group"] <- "EastEurope"
ref_list[grep("Rus", ref_list$name_prism_origin), 
         "origin_group"] <- "Russian"
ref_list[grep("German", ref_list$name_prism_origin), 
         "origin_group"] <- "German"
ref_list[grep("Ital", ref_list$name_prism_origin), 
         "origin_group"] <- "Italian"
ref_list[grep("Fren", ref_list$name_prism_origin), 
         "origin_group"] <- "French"
ref_list[grep("Celtic", ref_list$name_prism_origin), 
         "origin_group"] <- "AngloSaxon"
ref_list[grep("Nordic", ref_list$name_prism_origin), 
         "origin_group"] <- "Scandinavian"
ref_list[grep("Muslim", ref_list$name_prism_origin), 
         "origin_group"] <- "MiddleEast"
ref_list[grep("Persian", ref_list$name_prism_origin), 
         "origin_group"] <- "Persian"
ref_list[grep("Turkey", ref_list$name_prism_origin), 
         "origin_group"] <- "Turkey"
ref_list[grep("EastAsia", ref_list$name_prism_origin), 
         "origin_group"] <- "SouthEastAsia"
ref_list[grep("Chin", ref_list$name_prism_origin), 
         "origin_group"] <- "China"
ref_list[grep("Jap", ref_list$name_prism_origin), 
         "origin_group"] <- "Japan"
ref_list[grep("Kore", ref_list$name_prism_origin), 
         "origin_group"] <- "Korea"
ref_list[grep("Hisp", ref_list$name_prism_origin), 
         "origin_group"] <- "HispanicLatinAmerica"
ref_list[grep("Phil", ref_list$name_prism_origin), 
         "origin_group"] <- "Philippines"
ref_list[grep("SouthAsia", ref_list$name_prism_origin), 
         "origin_group"] <- "India"
ref_list[grep("Jewish", ref_list$name_prism_origin), 
         "origin_group"] <- "Jewish"
ref_list[grep("Greek", ref_list$name_prism_origin), 
         "origin_group"] <- "Greek"
ref_list[ref_list$origin_group %in% c("Russian", "EastEurope"), "origin_group"] <- "Russian&EastEurope"

################################################################
############## Calculate prediction probabilities ##############
################################################################

# prepare for classification
N = nrow(df)
n_X <- length(unique(ref_list$origin_group))+1
origin_groups <- as.data.frame(matrix(rep(NA, N * n_X), nrow = N))
colnames(origin_groups) <- c(colnames(df)[1], 
                             unique(ref_list$origin_group))
origin_groups[, 1] <- df[, 1]

## sum the corresponding probabilities together --------------------------------
for(i in 2:ncol(origin_groups)){
  origin_group <- colnames(origin_groups)[i]
  origins <- ref_list[ref_list$origin_group == origin_group, "name_prism_origin"]
  
  if(is.data.frame(df[, origins]) == FALSE){
    origin_prob <- df[, origins]}else{
      origin_prob <- rowSums(df[, origins])}
  
  origin_groups[, origin_group] <- origin_prob
}

df <- origin_groups
paste0("Custom class probabilities calculated for ", nrow(df), " observations")

##################################################
############## Construct indicators ##############
##################################################

# get maximum prediction
df$max_pred <- sapply(seq(1, nrow(df)), function(i){
  max_pred <- as.numeric(df[i, 2:ncol(df)])
  max_pred <- max(max_pred, na.rm = TRUE)
  return(max_pred)
})

# calculate distance to second highest prediction
dist_2nd <- function(dat){
  tmp <- t(dat[, !colnames(dat) %in% c("full_name", "max_pred")])
  tmp <- as.data.frame(tmp)
  tmp <- data.frame(sapply(tmp, as.numeric))
  tmp <- sapply(tmp, function(x) sort(x, decreasing = TRUE))
  tmp <- as.data.frame(tmp[1:2, ])
  distance_2nd <- sapply(tmp, function(x)abs(diff(x)))
  dat$distance_2nd <- distance_2nd
  return(dat)}
df <- dist_2nd(df)

# calculate entropy
entropy_fun <- function(dat){
  tmp <- t(dat[, !colnames(dat) %in% c("full_name", "max_pred", "distance_2nd")])
  tmp <- as.data.frame(tmp)
  tmp <- data.frame(sapply(tmp, as.numeric))
  entro <- sapply(tmp, function(x){
    entro <- x * log(x)
    entro <- - sum(entro)
    return(entro)}
  )
  dat$entropy <- entro
  return(dat)
}
df <- entropy_fun(df)


###########################################################
############## Estimate classification model ##############
###########################################################

