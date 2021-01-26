#######################################################################
# Description:    Script to classify and enocode retrieved names      #
#                 from nameprism for training classification models   #
# Authors:        Matthias Niggli/CIEB UniBasel                       #
# Date:           26.10.2020                                          #
#######################################################################

#######################################
## Load packages and set directories ##
#######################################

# packages for data processing: ------------------------------------------------
library("tidyverse")
library("stringi")

# directories  -----------------------------------------------------------------
mainDir1 <- "/scicore/home/weder/GROUP/Innovation/01_patent_data"

###################################
############ Load data ############
###################################
athlete_sample <- read.csv("Data/athlete_sample.csv") %>% rename(full_name = Name) %>% select(-Year)
inventor_sample <- readRDS(paste0(mainDir1, "/created data/origin_training_sample.rds"))
print("Data is loaded")

######################################################
############ Create custom country groups ############
######################################################

name_prism_origin <- names(inventor_sample)[3:41]
ref_list <- data.frame(name_prism_origin = name_prism_origin,
                       origin_group = NA)

origins <- c("Balkans", "Arabic", "Italian", "EastEurope",
             "Africa", "Italian", "Africa","French", "SouthEastAsia",
             "SouthEastAsia", "Jewish", "CentralAsia", "SouthEastAsia",
             "Scandinavian", "SouthEastAsia", "Scandinavian", "Persian", 
             "Scandinavian", "Arabic", "Greek", 
             "Arabic", "Hispanic", "Slavic", "Arabic", "Africa", "Japan",
             "German", "China", "India", "Hispanic", "Scandinavian", "Turkey",
             "Philippines", "AngloSaxon", "SouthEastAsia", "Korea", "Africa",
             "EastEurope", "SouthEastAsia")
ref_list$origin_group <- origins

# prepare for classification
N = nrow(inventor_sample)
n_X <- length(unique(ref_list$origin_group))+1
origin_groups <- as.data.frame(matrix(rep(NA, N * n_X), nrow = N))
colnames(origin_groups) <- c(colnames(inventor_sample)[1], 
                             unique(ref_list$origin_group))
origin_groups[, 1] <- inventor_sample[, 1]

## sum the corresponding probabilities together --------------------------------
for(i in 2:ncol(origin_groups)){
        origin_group <- colnames(origin_groups)[i]
        origins <- ref_list[ref_list$origin_group == origin_group, "name_prism_origin"]
        
        if(is.data.frame(inventor_sample[, origins]) == FALSE){
                origin_prob <- inventor_sample[, origins]}else{
                        origin_prob <- rowSums(inventor_sample[, origins])}
        
        origin_groups[, origin_group] <- origin_prob
}

inventor_sample <- origin_groups
paste0("Custom class probabilities calculated for ", nrow(inventor_sample), " observations")

#########################################################
############# Evaluate, subset and classify #############
#########################################################

## highest class probability
inventor_sample$max_pred <- unlist(lapply(seq(1, nrow(inventor_sample)), function(i){
  max_pred <- as.numeric(inventor_sample[i, 2:ncol(inventor_sample)])
  max_pred <- max(max_pred, na.rm = TRUE)
  return(max_pred)}))
# max_name <- function(dat){
#   tmp <- t(dat)[-1, ]
#   tmp <- as.data.frame(tmp)
#   tmp <- data.frame(sapply(tmp, as.numeric))
#   max_pred <- sapply(tmp, function(x) max(x, na.rm = TRUE))
#   dat$max_pred <- max_pred
#   return(dat)
# }
# inventor_sample <- max_name(inventor_sample)

## distance to 2nd highest class probability
dist_2nd <- function(dat){
  tmp <- t(dat[, !colnames(dat) %in% c("full_name", "max_pred")])
  tmp <- as.data.frame(tmp)
  tmp <- data.frame(sapply(tmp, as.numeric))
  tmp <- sapply(tmp, function(x) sort(x, decreasing = TRUE))
  tmp <- as.data.frame(tmp[1:2, ])
  distance_2nd <- sapply(tmp, function(x)abs(diff(x)))
  dat$distance_2nd <- distance_2nd
  return(dat)
}
inventor_sample <- dist_2nd(inventor_sample)

## entropy
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
inventor_sample <- entropy_fun(inventor_sample)

## subset to clearly identified names: -----------------------------------------
# HERE I COULD ALSO TRY OUT DIFFERENT COMBINATIONS OF PARAMETERS AND THEN ESTIMATE THE MODELS WITH
# THESE DIFFERNET DATASETS. USE THE DATASET WITH THE BEST CLASSIFICATION PERFORMANCE

MIN_PRED <- 0.6
MIN_DIST <- 0.2
MAX_ENTROPY <- 2

inventor_sample <- inventor_sample %>% filter(max_pred >= MIN_PRED,
                                              distance_2nd >= MIN_DIST,
                                              entropy <= MAX_ENTROPY)

paste0("Sample cleaned from ambigous origin classifications. Dropped ", 
       round(100 * (1-(nrow(inventor_sample)/N)), 2), "% of oberservations")

## classify "origin" to the maximum value --------------------------------------
inventor_sample$origin <- unlist(lapply(seq(1, nrow(inventor_sample)), function(i){
        tmp <- as.numeric(inventor_sample[i, 2:(ncol(inventor_sample)-3)])
        max_origin <- which(tmp == inventor_sample[i, "max_pred"])
        origin <- gsub("," ,"_", names(inventor_sample)[1+max_origin])
        return(origin)}
        ))

paste0("Training data ready. ", nrow(inventor_sample),
       " names classified to the highest origin probability")

############################################
############ BALANCE THE SAMPLE ############
############################################

## Drop, up-/down-sample classes in the training data --------------------------------------------
inventor_sample <- select(inventor_sample, full_name, origin)
inventor_sample %>% group_by(origin) %>% summarise(count = n(),
                                                   share = count / nrow(inventor_sample)) %>% View()
paste0("training sample ready: ", nrow(df_train), " additional observations for training.")

#####################################################
############ CREATE A CHARACTER DICTIONARY ##########
#####################################################

# double-check: search for special characters ----------------------------------
special_chars <- stri_extract_all(str = inventor_sample$full_name, regex = "[^a-z]")
special_chars <- unlist(special_chars)
special_chars <- unique(special_chars)
if(special_chars == " "){print("no special chars left")}else{warning("Special characters in sample")}

## choose character dictionary -------------------------------------------------
char_dict <- NULL
for(i in 1:length(inventor_sample$full_name)){
        chars <- unlist(str_extract_all(inventor_sample$full_name[i], "[:print:]"))
        chars <- unique(chars)
        char_dict <- c(char_dict, chars[!chars %in% char_dict])
}
char_dict <- c(sort(char_dict)[-1], " ","END")
if(length(letters %in% char_dict) < 26){warning("Not all standard letters are included in the vocabulary")}else{
        print("All standard letters are included in the vocabulary")}
n_chars <- length(char_dict)

# choose sequence length -------------------------------------------------------
hist(nchar(inventor_sample$full_name), main = "", 
     xlab = "Number of characters per name") # highlight name length distribution
seq_tresh <- 30
paste0("truncating ", length(tmp[tmp>seq_tresh]), " (", 
       round(100-length(tmp[tmp<=seq_tresh]) / nrow(inventor_sample) *100, 2),
       "%) names to ", seq_tresh, " characters")
max_char <- seq_tresh

PARAMS <- data.frame(SEQ_MAX = max_char, N_CHARS = n_chars)
CHAR_DICT <- char_dict

####################################
######### SAVE THE DATA # ##########
####################################

write.csv(x = inventor_sample, 
          file = paste0(getwd(), "/Data/inventor_sample.csv"),
          row.names = FALSE)
write.csv(x = char_dict, file = paste0(getwd(), "/Data/CHAR_DICT.csv"),
          row.names = FALSE)
write.csv(x = PARAMS, file = paste0(getwd(), "/Data/PARAMS.csv"),
          row.names = FALSE)
