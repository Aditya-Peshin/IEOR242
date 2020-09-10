# Homework 5 

setwd("D:\\Studies\\Berkeley\\Semester 1\\242\\Hw 5")

library(softImpute)
library(randomForest)
library(ranger)
library(dplyr)
library(tidyverse)
library(reshape2)
library(glmnet)
library(broom)
library(rpart)

OSR2 <- function(predictions, train, test) {
  SSE <- sum((test - predictions)^2)
  SST <- sum((test - mean(train))^2)
  r2 <- 1 - SSE/SST
  return(r2)
}

musicratings <- read.csv("D:\\Studies\\Berkeley\\Semester 1\\242\\Hw 5\\MusicRatings.csv")
songs <- read.csv("D:\\Studies\\Berkeley\\Semester 1\\242\\Hw 5\\Songs.csv")
users <- read.csv("D:\\Studies\\Berkeley\\Semester 1\\242\\Hw 5\\Users.csv")

# PART (A)
str(users)
# 2421 users

str(songs)
# 807 songs

summary(musicratings$rating)
# Range of values of scores are [1,3.433]

# Splitting the data into training and testing (8%)
set.seed(345)
train.ids <- sample(nrow(musicratings), 0.92*nrow(musicratings))
train <- musicratings[train.ids,]
test <- musicratings[-train.ids,]

# split training into real training (84%) and 2 validation sets
# 4% for Collaborative Filtering
val1.ids <- sample(nrow(train), (4/92)*nrow(train))
val1 <- train[val1.ids,]
train <- train[-val1.ids,]

# 4% for Blending
val2.ids <- sample(nrow(train), (4/88)*nrow(train))
val2 <- train[val2.ids,]
train <- train[-val2.ids,]

# Constructing an Incomplete Training set ratings matrix
mat.train <- Incomplete(train$userID, train$songID, train$rating)
summary(train)

# PART (B)
# PART (B.i)
# Here, we will need 2421(nusers) + 807(nsongs) = 3228 parameters to be trained
set.seed(345)
mat.train.centered <- biScale(mat.train, maxit = 1000, row.scale = FALSE, col.scale = FALSE)


# PART (B.ii)
# mat.train.centered is X_ij - alpha_i - beta_j
alpha <- attr(mat.train.centered, "biScale:row")$center
beta <- attr(mat.train.centered, "biScale:column")$center

# Reordering the songs matrix in order to find the most popular songs
songs_new <- songs
songs_new$BETA <- beta
songs_new <- songs_new[order(-songs_new$BETA),]
head(songs_new,3)
# songID       songName year          artist   genre     BETA
# 54      54 You're The One 1990   Dwight Yoakam Country 1.710318
# 26      26           Undo 2001           Bjork    Rock 1.691155
# 439    439        Secrets 2009     OneRepublic    Rock 1.641377

# PART (B.iii)
# Reordering the users matrix to see which are the most enthused about music
users_new <- users
users_new$ALPHA <- alpha
users_new <- users_new[order(-users_new$ALPHA),]
head(users_new,3)
# userID     ALPHA
# 1540   1540 0.5959218
# 838     838 0.4964753
# 1569   1569 0.4770457

# PART (B.iv)
# Out of Sample Performance of Simple Additive Model
# Creating matrix of prediction values
SAMpreds <- matrix(rep(NA,1953747), nrow = 2421, ncol = 807)
for(i in nrow(SAMpreds))
  for(j in ncol(SAMpreds))
    SAMpreds[i,j] = alpha[i] + beta[j]

str(test)
# MAE of the Simple Additive Model
mean(abs(SAMpreds[test$userID,test$songID] - test$rating))/5
# MAE = 0.04084026

# RMSE of the Simple Additive Model
sqrt(mean((SAMpreds[test$userID,test$songID] - test$rating)^2))/5
# RMSE = 0.05582044

# R^2 of the Simple Additive Model
# I use a copy of the test set and add Alpha and Beta columns to it
testa <- test
testa$alpha <- alpha[testa$userID]
testa$beta <- beta[testa$songID]
# Prediction = alpha[i] + beta[i]
testa$preds <- testa$alpha + testa$beta
# R2
OSR2(testa$preds, train$rating, testa$rating)
# R^2 = 0.2799879

# PART (C) - Collaborative Filtering

# Part (C.i)
# The number of parameters in this model will be 2421 + 807 + W + S 
# W - > Weights of each obs wrt archetype i.e. 2421 * k no of weights
# S -> Archetype ratings for each movie i.e. 807 *k

# Part (C.ii)
# Trying different number of archetypes, i.e. rank = 1,2,...,20
CF.mae.vals = rep(NA, 20)
for (rnk in seq_len(20)) {
  print(str_c("Trying rank.max = ", rnk))
  CFmod <- softImpute(mat.train.centered, rank.max = rnk, lambda = 0, maxit = 1000)
  preds <- impute(CFmod, val1$userID, val1$songID, unscale = TRUE) %>% pmin(5) %>% pmax(1)
# here the impute function has a parameter called "unscale" which is set to true
# This parameter, when set to true, reverses the centering and scaling on the predictions. 
  CF.mae.vals[rnk] <- mean(abs(preds - val1$rating))
}

CF.mae.val.df <- data.frame(rnk = seq_len(20), mae = CF.mae.vals)
ggplot(CF.mae.val.df, aes(x = rnk, y = mae)) + geom_point(size = 3) + 
  ylab("Validation MAE") + xlab("Number of Archetypal Users") + 
  theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18))


# Final value of k = 5 from graph

# Part (C.iii)
set.seed(345)
finalCFmod <- softImpute(mat.train.centered, rank.max = 5, lambda = 0, maxit = 1000)
finalCFpreds <- impute(finalCFmod, test$userID, test$songID) %>% pmin(5) %>% pmax(1)

# Calculating the MAE
mean(abs(finalCFpreds - test$rating))/5
# Here, the MAE is 0.03464215

# Calculating the RSS
sqrt(mean((finalCFpreds - test$rating)^2))/5
# Here, the RMSE is 0.04733791 

# Calculating R^2
OSR2(finalCFpreds, train$rating, test$rating)
# Here, the R^2 is 0.2891153

# PART (D)
# Part (D.i)

# Building a model using only genre of the song
# Trying linear regression
train2 <- train
train2$genre <- songs$genre[train2$songID]
train2$year <- songs$year[train2$songID]
test2 <- test
test2$genre <- songs$genre[test2$songID]
test2$year <- songs$year[test2$songID]
val2a <- val2
val2a$genre <- songs$genre[val2a$songID]
val2a$year <- songs$year[val2a$songID]

modLM <- lm(rating ~ genre, data = train2)
summary(modLM)
modLMpreds <- predict(modLM, newdata=test2)

# Calculating the MAE
mean(abs(modLMpreds - test2$rating))/5
# Here, the MAE is 0.0459205

# Calculating the RSS
sqrt(mean((modLMpreds - test2$rating)^2))/5
# Here, the RMSE is 0.05583358 

# Calculating R^2
OSR2(modLMpreds, train2$rating, test2$rating)
# OSR2 = 0.01105538 

# Building a model using only the year the song was released
set.seed(456)
loss.mat <- cbind(c(0, 20), c(1, 0))

modLM2 <- randomForest(rating ~ year, data = train2, mtry = 1, ntree = 500)
summary(modLM2)
modLM2preds <- predict(modLM2, newdata=test2)

# Calculating the MAE
mean(abs(modLM2preds - test2$rating))/5
# Here, the MAE is 0.04631903

# Calculating the RSS
sqrt(mean((modLM2preds - test2$rating)^2))/5
# Here, the RMSE is 0.05608735 

# Calculating R^2
OSR2(modLM2preds, train2$rating, test2$rating)
# OSR2 = 0.002045203

# Part (D.ii)
# Using Validation set 2 to build blending model

val2predsCF <- impute(finalCFmod, val2a$userID, val2a$songID) %>% pmin(5) %>% pmax(1)
val2predsLMG <- predict(modLM, newdata=val2a)
val2predsLMY <- predict(modLM2, newdata=val2a)
val.blending_df = data.frame(rating = val2a$rating, cf_preds = val2predsCF, lm1_preds = val2predsLMG , lm2_preds =val2predsLMY)

mod_blend = lm(rating ~ . -1, data = val.blending_df)
summary(mod_blend)

test2CF  <- impute(finalCFmod, test2$userID, test2$songID) %>% pmin(5) %>% pmax(1)
test2predsLMG <- predict(modLM, newdata=test2)
test2predsLMY <- predict(modLM2, newdata=test2)
test.blending_df = data.frame(rating = test2$rating, cf_preds = test2CF, lm1_preds = test2predsLMG , lm2_preds =test2predsLMY)

test.preds.blend <- predict(mod_blend, newdata = test.blending_df)

mean(abs(test.preds.blend - test$rating))/4
# MAE = 0.04432202

sqrt(mean((test.preds.blend - test$rating)^2))/4
# RMSE = 0.05819273

OSR2(test.preds.blend, train$rating, test$rating)
# OSR2 = 0.3124592














# SSE_SAM = 0
# SST_SAM = 0
# mean_rating = mean(train$rating)
# for (i in test$userID){
#   print(i)
#   for(j in test$songID){
#     SSE_SAM = SSE_SAM + (test$rating[test$userID == i && test$songID == j] -  SAMpreds[i,j])^2
#     SST_SAM = SST_SAM + (test$rating -  mean_rating)^2
#   }
# }
# SSE_SAM <- sum((test$rating - SAMpreds[test$userID,test$songID]$)^2)
# SST_SAM <- sum((test$rating - mean(train$rating))^2)
# R2_SAM <- 1 - SSE_SAM/SST_SAM
# R2_SAM
# 
# Value =


