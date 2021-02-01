library("tidyverse")
library("data.table")
library("stringi")
library("countrycode")

mainDir1 <- "/scicore/home/weder/GROUP/Innovation/01_patent_data"

# names retrieved from NamePrism: ----------------------------------------------
inv_names_origin <- readRDS(paste0(mainDir1,
                                   "/created data/origin_training_sample.rds"))

# all inventors data -----------------------------------------------------------
inv_reg <- readRDS(paste0(mainDir1, "/created data/inv_reg.rds"))
name_origin <- inv_reg %>% select(name, Ctry_code, p_year) %>% 
  distinct(name, .keep_all = TRUE) %>% na.omit()

# clean names
name_origin$name <- tolower(name_origin$name)
name_origin$name <- gsub("[[:punct:]]", "", name_origin$name)
name_origin$name <- gsub("[0-9]", "", name_origin$name)

# find names with special characters from sample
sc_idx <- grep("[^a-z]", gsub(pattern = " ", replacement = "", name_origin$name))
special_chars <- gsub(" ", replacement = "", name_origin$name[sc_idx])
special_chars <- stri_extract_all(str = special_chars, regex = "[^a-z]")
special_chars <- unlist(special_chars)
special_chars <- unique(special_chars)
special_chars <- special_chars[is.na(special_chars) == FALSE]
special_chars
repl_vec <- iconv(special_chars, to = "ASCII//TRANSLIT")
manual_change <- which(repl_vec == "?")
special_chars[manual_change]
repl_vec[manual_change] <- c("o", "a", "b", "t", "th", "th", "l", "e", "th")
source("/scicore/home/weder/nigmat01/inventor_migration/Code/training_data/clear_chars_function.R")
name_origin$name <- clear_chars(name_origin$name, special_chars = special_chars,
                                repl_vec = repl_vec)
print("All special characters removed from names")

# merge together: ---------------------------------------------------------------------
name_origin <- rename(name_origin, full_name = name)
name_origin <- setDT(name_origin)[full_name %in% inv_names_origin$full_name, ]
inv_names_origin <- merge(inv_names_origin, name_origin, by = "full_name", all.x = TRUE)
countries <- inv_names_origin %>% group_by(Ctry_code) %>% summarize(count = n())
countries$Ctry_code <- ifelse(countries$count < 50, "Other", countries$Ctry_code)
countrycode(sourcevar = unique(countries$Ctry_code)[-1],
            origin = "iso2c", destination = "country.name.en")

# => cannot reconstruct the data completly... matches only for around 63000 of the 66000 inventors
# listed in the sample. either drop the non-matched or let it slide..
