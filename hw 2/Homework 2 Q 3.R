# Homework 2 Q 3

# loading required libraries
library(dplyr)
library(ggplot2)
library(caTools)
library(GGally)
library(ROCR)

# reading in the file
framingham <- read.csv("framingham.csv")
str(framingham)

# setting the factor variables as factors
framingham$TenYearCHD <- as.factor(framingham$TenYearCHD)
framingham$male <- as.factor(framingham$male)
framingham$currentSmoker <- as.factor(framingham$currentSmoker)
framingham$BPMeds <- as.factor(framingham$BPMeds)
framingham$prevalentStroke <- as.factor(framingham$prevalentStroke)
framingham$prevalentHyp <- as.factor(framingham$prevalentHyp)
framingham$diabetes <- as.factor(framingham$diabetes)
framingham$education <- as.factor(framingham$education)

# splitting the data into training and testing
set.seed(144)
split <- sample.split(framingham$TenYearCHD, 0.7)

framingham.train <- filter(framingham, split == TRUE)
framingham.test <- filter(framingham, split == FALSE)

# using logistic regression to make model
mod1 <- glm(TenYearCHD ~ . , data = framingham.train, family = binomial)
summary(mod1)

# making a prediction on the test data
pred_CHD = predict(mod1 , newdata = framingham.test, type = "response")

# confusion matrix for training data
table (framingham.test$TenYearCHD, pred_CHD>= 0.16)

# confusion matrix for baseline model
table (framingham.test$TenYearCHD, pred_CHD>= 1)

# Probability that the new patient has the disease 10 years from now
new_patient <- data.frame(male = '0', age = 51, education = "College", currentSmoker = '1', cigsPerDay = 20, BPMeds = '0', prevalentStroke = '0',prevalentHyp = '1', diabetes = '0', totChol = 220, sysBP = 140, diaBP = 100, BMI = 31, heartRate = 59, glucose = 78)                    
predict(mod1, new_patient, type = "response")

# Plotting a ROC curve for the model
pred <- prediction(pred_CHD, framingham.test$TenYearCHD)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col = "Blue")
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]    
auc     # to print value of AUC

