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

# Read in experiment one data
total <- read.csv("../aws_logs/combined.csv")
total$Type <- as.factor(total$Type)
total$Grammar <- as.factor(total$Grammar)

total_ppa <- total %>% filter(Grammar == "PPA")
total_sc <- total %>% filter(Grammar == "SC")

ggplot(total_ppa, aes(Data, Cosine)) + geom_line(aes(color=Model))+ facet_grid(~Hidden) + ggtitle("RAAM vs DRAAM vs DRAAM+ (PPA)")
ggplot(total_sc, aes(Data, Cosine)) + geom_line(aes(color=Model))+ facet_grid(~Hidden) + ggtitle("RAAM vs DRAAM vs DRAAM+ (SC)")

# Read in dropout data
dropout <- read.csv("../aws_logs/dropout.csv")
dropout <- dropout %>% filter(Hidden == "400" & Data == "2000")

ggplot(dropout, aes(Dropout, Cosine)) + geom_line(aes(color=Grammar))+ ggtitle("DRAAM+ Dropout")

losses <- read.csv("../aws_logs/training_losses.csv")

ggplot(losses, aes(Epoch, Loss)) + geom_line(aes(color=Model)) + ggtitle("RAAM vs DRAAM")


