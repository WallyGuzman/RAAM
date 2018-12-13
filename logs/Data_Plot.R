# Plotting code for 394N Final Project

library(ggplot2)
library(dplyr)

# Load data files
one_hot <- read.csv("../logs/results_one_hot.csv")
draam <- read.csv("../logs/results_draam.csv")

# Add model names and drop NaNs
one_hot$Model <- "RAAM"
draam$Model <- "DRAAM"

# Combine and turn into data frame
combined <- data.frame(rbind(one_hot, draam))
combined$Type <- as.factor(combined$Type)

# Split based on data type
combined_ppa <- combined %>% filter(Test == "PPA")
combined_sc <- combined %>% filter(Test == "SC")

# Plot
ggplot(combined_ppa, aes(Data, Cosine)) + geom_line(aes(color=Type)) + facet_grid(~Hidden*Model) + ggtitle("DRAAM vs RAAM (PPA)")
ggplot(combined_sc, aes(Data, Cosine)) + geom_line(aes(color=Type)) + facet_grid(~Hidden*Model) + ggtitle("DRAAM vs RAAM (SC)")
