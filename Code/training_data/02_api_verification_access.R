#######################################################################
# Description:    Script to retrieve name-prism predictions for a     #
#                 sample consisting of olympic athletes name's which  #
#                 are labeled to ethnic origins.                      #
# Authors:        Matthias Niggli/CIEB UniBasel                       #
# Date:           17.02.2021                                          #
#######################################################################

#######################################
## Load packages and set directories ##
#######################################

# packages for data processing: ------------------------------------------------
library("tidyverse")

# packages for prism-name API: -------------------------------------------------
library("jsonlite")

# directories and reproducibility
if(substr(x = getwd(), 
          nchar(getwd())-13, nchar(getwd())) == "name_to_origin"){
  print("Working directory corresponds to repository directory")}else{
    print("Make sure your working directory is the repository directory.")}
set.seed(24012021)

#########################
## Load & process data ##
#########################

df_olympic <- read.csv("Data/athlete_sample.csv")

## sample names for name-prism -------
test_sample <- sample_n(df_olympic, size = 3000)
test_sample$full_name_encoded <- gsub(" ", "%20", x = test_sample$Name)

###########################
## Access name-prism API ##
###########################

## specify the API-Token and URL -----------------------------------------------
API_nameprism <- "8cf1f0d395a1daac"
pred_type = "nat"
response_format = "json"

# get the 39 different origins from the API
for(i in 5:43){test_sample[, i] <- NA}
api_url <- paste("http://www.name-prism.com/api_token/",
                 pred_type, "/", response_format, "/",
                 API_nameprism, "/",
                 "test%20name", sep = "")
origins <- names(fromJSON(txt = api_url))
names(test_sample)[5:43] <- origins
print("Prepared dataframe to store origin predictions from name-prism.")

## NOT RUN: Get name-prism predictions for all names

for(i in 1:nrow(test_sample)){
  api_url <- paste0("http://www.name-prism.com/api_token/",
                    pred_type, "/", response_format, "/",
                    API_nameprism, "/", test_sample$full_name_encoded[i])
  tmp <- fromJSON(txt = api_url)
  tmp <- unlist(tmp)
  test_sample[i, 5:43] <- tmp
  Sys.sleep(0.35) # without Sys.sleep: ~0.5s per loop
}
print(paste("Origin predictions for", nrow(test_sample), "names retrieved"))

###############################
## Save the verification set ##
###############################

write.csv(x = test_sample, file = "Data/API_verification_sample.csv", row.names = FALSE)



