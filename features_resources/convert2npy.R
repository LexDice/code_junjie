library(reticulate)

np <- import("numpy")
load("Autism_Data.rdata")

neg_cases <- Autism_Data$Group_1
pos_cases <- Autism_Data$Group_2

neg_arr <- np$array(neg_cases)
pos_arr <- np$array(pos_cases)

np$save("neg_features.npy", neg_arr)
np$save("pos_features.npy", pos_cases)
