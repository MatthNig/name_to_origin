
# names in the original sample with country of residence information:
datDir <- "/scicore/home/weder/GROUP/Innovation/01_patent_data"
inv_names <- readRDS(paste0(datDir,
                            "/created data/origin_name_sample.rds"))

# names scrapped from name-prism:
inv_names_origin <- readRDS(paste0(datDir,
                            "/created data/origin_training_sample.rds"))
inv_names_origin <- merge(inv_names_origin, inv_names[, 1:3], by = "full_name", all.x = TRUE)
inv_names_origin %>% group_by(ctry_group) %>% summarize(count = n()) %>% View()
inv_names_origin %>% filter(is.na(Ctry_code) == FALSE) %>% distinct(Ctry_code)
# what I don't know is how many SEA and Turkish observations I have. But these can be inferred from balancing script 