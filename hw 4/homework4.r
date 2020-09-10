# Homework 4 

setwd("C:\\Users\\Aditya Peshin\\Desktop\\Studies\\UC Berkeley\\Fall 2019\\242 - Applications in Data Analysis\\Assignments\\hw 4")
getwd()
library(tm)
library(SnowballC)
library(wordcloud)
library(MASS)
library(caTools)
library(dplyr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(tm.plugin.webmining)
library(corpus)
library(boot)
library(ROCR)
library(GGally)

# Defining the functions to calculate the metrics

tableAccuracy <- function(test, pred) {
  t = table(test, pred)
  a = sum(diag(t))/length(test)
  return(a)
}

tableTPR <- function(label, pred) {
  t = table(label, pred)
  return(t[2,2]/(t[2,1] + t[2,2]))
}

tableFPR <- function(label, pred) {
  t = table(label, pred)
  return(t[1,2]/(t[1,1] + t[1,2]))
}

tablePrec <- function(label, pred) {
  t = table(label, pred)
  return(t[2,2]/(t[1,2] + t[2,2]))
}

boot_accuracy <- function(data, index) {
  labels <- data$response[index]
  predictions <- data$prediction[index]
  return(tableAccuracy(labels, predictions))
}

boot_tpr <- function(data, index) {
  labels <- data$response[index]
  predictions <- data$prediction[index]
  return(tableTPR(labels, predictions))
}

boot_fpr <- function(data, index) {
  labels <- data$response[index]
  predictions <- data$prediction[index]
  return(tableFPR(labels, predictions))
}

boot_all_metrics <- function(data, index) {
  acc = boot_accuracy(data, index)
  tpr = boot_tpr(data, index)
  fpr = boot_fpr(data, index)
  return(c(acc, tpr, fpr))
}
# Load file into R, create new column for predictor variable
stack_orig <- read.csv("sover.csv", stringsAsFactors = FALSE)
stack_orig$BinarySc <- ifelse(stack_orig$Score>=1, 1, 0)
str(stack_orig)

# Converting to factor variable
stack_orig$BinarySc <- as.factor(stack_orig$BinarySc)
st2 <- stack_orig
st2$Score <- NULL

v1 = Corpus(VectorSource(st2$Title))
v2 = Corpus(VectorSource(st2$Body))

j = 0
length(v2)

for (i in 1:length(v2)) {
  v2[[i]][["content"]] = extractHTMLStrip(v2[[i]][["content"]])
  print(j)
  j = j+1
}

# Changing text to lower case
modv1 = tm_map(v1, tolower)
modv2 = tm_map(v2, tolower)
  
strwrap(modv1[["1"]])
strwrap(modv2[["1"]])

# Removing \n from text
modv1 = tm_map(modv1, removeWords, "\n")
modv2 = tm_map(modv2, removeWords, "\n")

strwrap(modv1[["1"]])
strwrap(modv2[["1"]])

# Removing Punctuation
modv1 = tm_map(modv1, removePunctuation)
modv2 = tm_map(modv2, removePunctuation)  

strwrap(modv1[["1"]])
strwrap(modv2[["1"]])

# Removing Stop Words

# stopwords("english")[1:10]
# length(stopwords("english"))
modv1 = tm_map(modv1, removeWords, stopwords("english"))
modv2 = tm_map(modv2, removeWords, stopwords("english"))

strwrap(modv1[["1"]])
strwrap(modv2[["1"]])

# Stemming the Document

modv1 = tm_map(modv1, stemDocument)
modv2 = tm_map(modv2, stemDocument)  

strwrap(modv1[["1"]])
strwrap(modv2[["1"]])

# Frequencies of words appearing

f1 = DocumentTermMatrix(modv1)
f2 = DocumentTermMatrix(modv2)

# For me to see which teerms are the most common
findFreqTerms(f1, lowfreq=450)
findFreqTerms(f2, lowfreq=2800)

# Identifying number of sparse terms
sparse1 = removeSparseTerms(f1, 0.92)
sparse2 = removeSparseTerms(f2, 0.835)

# Creating Document Term Matrix

Title_TM <- as.data.frame(as.matrix(sparse1))
colnames(Title_TM)
Body_TM <- as.data.frame(as.matrix(sparse2))
colnames(Body_TM)

colnames(Title_TM) <- paste0("Title" , sep = '_' , colnames(Title_TM) )
colnames(Title_TM)
colnames(Body_TM) <- paste0("Body", sep = "_", colnames(Body_TM))
colnames(Body_TM)

Combined_TM <- cbind(Title_TM, Body_TM)
colnames(Combined_TM)
nrow(Combined_TM)
Combined_TM$BinarySc <-  st2$BinarySc

# Splitting data into training and test sets

set.seed(123)
spl = sample.split(Combined_TM$BinarySc, SplitRatio = 0.7)
  
Combined_TM.train <- filter(Combined_TM, spl == TRUE)
Combined_TM.test <- filter(Combined_TM, spl == FALSE)

# Logistic Regression

mod_glm <- glm(BinarySc~. , data = Combined_TM.train, family = "binomial")
summary(mod_glm)

PredictLog = predict(mod_glm, newdata = Combined_TM.test, type = "response")
table(Combined_TM.test$BinarySc, PredictLog > 0.5)
#   FALSE TRUE
# 0   750  387
# 1   550  553
tableAccuracy(Combined_TM.test$BinarySc, PredictLog > 0.5)
# 0.5816964

# LDA

mod_lda = lda(BinarySc ~ ., data = Combined_TM.train)

PredictLDA = predict(mod_lda, newdata = Combined_TM.test)$class
table(Combined_TM.test$BinarySc, PredictLDA)
#     0   1
# 0 753 384
# 1 551 552
tableAccuracy(Combined_TM.test$BinarySc, PredictLDA)
# 0.5825893

# Cross Validated Random Forests

mod_basicRF = train(BinarySc ~ ., 
                    method = "rf",
                    data=Combined_TM.train,
                    tuneGrid = data.frame(mtry = 1:20),
                    trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE))
mod_basicRF$results
# Best value of mtry = 2
final_CVrf = mod_basicRF$finalModel

Predict_CVRF = predict(final_CVrf, newdata = Combined_TM.test)
table(Combined_TM.test$BinarySc, Predict_CVRF)
#     0   1
# 0 822 315
# 1 644 459
tableAccuracy(Combined_TM.test$BinarySc, Predict_CVRF)
# 0.571875

as.data.frame(final_CVrf$importance) %>%
  mutate(Words = rownames(final_CVrf$importance)) %>%
  arrange(desc(MeanDecreaseGini))

# Stepwise Logistic Regression

mod_StepLog = step(mod_glm, direction = "backward")
summary(mod_StepLog)
length(mod_StepLog$coefficients)  # 20 variables selected in the final model

PredictStepLog = predict(mod_StepLog, newdata = Combined_TM.test, type = "response")
table(Combined_TM.test$BinarySc, PredictStepLog > 0.5)
# FALSE TRUE
# 0   749  388
# 1   561  542
tableAccuracy(Combined_TM.test$BinarySc, PredictStepLog > 0.5)
# 0.5763393

# Basic Boosting

mod_boost <- train( BinarySc ~ .,
                     data = Combined_TM.train,
                     method = "gbm",
                     metric = "Accuracy",
                     distribution = "bernoulli")
mod_boost

final_boost <- mod_boost$finalModel

PredictBoost = predict(mod_boost, newdata = Combined_TM.test)
table(Combined_TM.test$BinarySc, PredictBoost)
#     0   1
# 0 700 437
# 1 659 444
tableAccuracy(Combined_TM.test$BinarySc, PredictBoost)
# 0.5107143

# Cross Validated Boosting

tGrid = expand.grid(n.trees = (1:25)*100, interaction.depth = c(1,2,4,8),
                    shrinkage = 0.1, n.minobsinnode = 10)

mod_CVboost <- train(BinarySc ~ .,
                     data = Combined_TM.train,
                     method = "gbm",
                     tuneGrid = tGrid,
                     trControl = trainControl(method="cv", number=5, verboseIter = TRUE),
                     metric = "Accuracy",
                     distribution = "bernoulli")
mod_CVboost
mod_CVboost$results

# Boosting Cross Validation Round 2
tGrid2 = expand.grid(n.trees = (1:20)*50, interaction.depth = c(1,2,3),
                    shrinkage = 0.01, n.minobsinnode = 10)

mod_CVboost2 <- train(BinarySc ~ .,
                     data = Combined_TM.train,
                     method = "gbm",
                     tuneGrid = tGrid2,
                     trControl = trainControl(method="cv", number=5, verboseIter = TRUE),
                     metric = "Accuracy",
                     distribution = "bernoulli")
mod_CVboost2
mod_CVboost2$results
final_CVboost <- mod_CVboost2$finalModel
Combined_TM.test.mm = as.data.frame(model.matrix(BinarySc ~ . +0, data = Combined_TM.test))
PredictCVBoost = predict(final_CVboost, newdata = Combined_TM.test.mm, n.trees = 650, type = "response")
table(Combined_TM.test$BinarySc, PredictCVBoost < 0.5)
#   FALSE TRUE
# 0   756  381
# 1   553  550
tableAccuracy(Combined_TM.test$BinarySc, PredictCVBoost < 0.5)
# 0.5830357


# Bootstrapping the test set for identifying variability of performance metrics

Boost_test_set = data.frame(response = Combined_TM.test$BinarySc, predictions = PredictCVBoost < 0.5)
set.seed(123)
Boost_boot = boot(Boost_test_set, boot_all_metrics, R = 10000)
Boost_boot
boot.ci(Boost_boot, index = 1, type = "basic")
boot.ci(Boost_boot, index = 2, type = "basic")
boot.ci(Boost_boot, index = 3, type = "basic")

# Calculating the performance of the model using precision 

# For the Cross Validated Boosting MOdel
CVboostpred <- prediction(1 - PredictCVBoost, Combined_TM.test$BinarySc)
CVBoostPerf <- performance(CVboostpred, "prec")
plot(CVBoostPerf, colorize = FALSE)
as.numeric(performance(CVboostpred, "auc")@y.values)

# For the Logistic Regression Model
logisticpred <- prediction(PredictLog, Combined_TM.test$BinarySc)
logisticPerf <- performance(logisticpred, "prec")
plot(logisticPerf, colorize = FALSE)
as.numeric(performance(logisticpred, "auc")@y.values)

# For the Stepwise Logistic Regression Model
Steplogpred <- prediction(PredictStepLog, Combined_TM.test$BinarySc)
SteplogPerf <- performance(Steplogpred, "prec")
plot(SteplogPerf, colorize = FALSE)
as.numeric(performance(Steplogpred, "auc")@y.values)

# Calculating the value of precision for each of the models at their optima

# CV Boosting
tablePrec(Combined_TM.test$BinarySc, PredictCVBoost < 0.6)
#0.40  -> 0.690 prec

# Logistic Regression
tablePrec(Combined_TM.test$BinarySc, PredictLog > 0.735)
#0.735 -> 0.774 prec

# Stepwise Logistic Regression
tablePrec(Combined_TM.test$BinarySc, PredictStepLog > 0.625)
#0.625 -> 0.6835 prec

table(Combined_TM.test$BinarySc)
