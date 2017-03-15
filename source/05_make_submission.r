
#---------------------------
# library
#---------------------------

library(dplyr)
library(data.table)
library(magrittr)

#---------------------------
# main
#---------------------------

base <- fread("../dataset/sample_submission.csv")
base_wkt <- base$MultipolygonWKT
base$MultipolygonWKT <- NULL

train_wkt <- fread("../dataset/train_wkt_v4.csv")
overlap <- intersect(unique(base$ImageId), unique(train_wkt$ImageId))

target_path <- "../submission/submission_ave/wkt_class_all_valid0.486.csv"

submit <- fread(target_path, data.table=FALSE)
colnames(submit) <- c("ImageId", "ClassType", "MultipolygonWKT")

result <- left_join(base, submit, by=c("ImageId", "ClassType"))

cat("num_na: ", sum(is.na(result)), "\n")

if(sum(is.na(result$MultipolygonWKT)) > 0){
	warning("Not all of WKT are found !")

	na_index <- is.na(result$MultipolygonWKT)
	result$MultipolygonWKT[na_index] <- "MULTIPOLYGON EMPTY"

	cat("num_na: ", sum(is.na(result)), "\n")
}

# replace overlap images ----------

for(img in overlap){
	cat("\treplace: ", img, "\n")
	result[result$ImageId == img,] <- train_wkt[train_wkt$ImageId == img,]
}

# check result ----------

stopifnot(all(base$ImageId == result$ImageId))
stopifnot(all(base$ClassType == result$ClassType))

# save result ----------

file_path <- paste0("../submission/submission_",
	gsub(".+(class.+).csv", "\\1", target_path), ".csv.gz")

handler <- gzfile(file_path)
write.csv(result, handler, row.names=FALSE, quote=3)
