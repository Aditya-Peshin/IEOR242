library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(gbm)
library(caTools)
library(dplyr)
library(ggplot2)
library(GGally)
library(ROCR)
library(e1071)
library(tidyverse)
library(MASS)

# Assignment 3 Question 2

# Part A
letters_orig <- read.csv("Letters.csv")
letters_orig$isB = as.factor(letters_orig$letter == "B")

#setting seed
set.seed(456)

#splitting into training and testing set
train.ids = sample(nrow(letters_orig), 0.65*nrow(letters_orig))
letters_orig.train = letters_orig[train.ids,]
letters_orig.test = letters_orig[-train.ids,]

# Part A.i [ Baseline Model ]
set.seed(456)
table (letters_orig.test$isB)
    # False True
    # 788   303
baseline_acc <- (788) / nrow(letters_orig.test)
baseline_acc  
    # This value is 0.7222

# Part A.ii [ Logistic Regression ]
set.seed(456)
mod1 <- glm(isB ~ . - letter, data = letters_orig.train, family = "binomial" )
summary(mod1)
    # making the prediction on the test data
pred1 <- predict(mod1, newdata = letters_orig.test, type = "response")
table(letters_orig.test$isB, pred1>= 0.5)
    #        FALSE TRUE
    # FALSE   760   28
    # TRUE     30  273
logistic_acc <- (273 + 760)/(273+760+30+28)
logistic_acc
    # this accuracy is 0.9468

# Part A.iii  [ Area under the Curve ]
set.seed(456)
pred_logistic <- prediction(pred1, letters_orig.test$isB)
auc_logistic <- performance(pred_logistic, measure = "auc")
auc_logistic <- auc_logistic@y.values[[1]]
    # The value of AUC is 0.97966

# Part A.iv   [ CART Tree to predict letter B]
set.seed(456)
cpVals = data.frame(cp = seq(0, .04, by=.002))
model_cart <- train(isB ~ . - letter, 
                   data = letters_orig.train, 
                   method = "rpart", 
                   tuneGrid = cpVals,
                   trControl = trainControl(method = "cv", number=10),
                   metric = "Accuracy" )
model_cart$results
model_cart
    # for cp = 0.004, we get maximum accuracy of 0.9323 on the training set

# Now, we predict the test set values and calculate test set accuracy
set.seed(456)
pred2 <- predict(model_cart$finalModel, newdata = letters_orig.test, type = "prob")
table(letters_orig.test$isB, pred2[,2] >= 0.5)
    #         FALSE TRUE
    # FALSE     768   20
    # TRUE       62  241
cart_acc <- (241 + 768)/(241+20+62+768)
cart_acc
    # this accuracy is 0.924839

# Part A.v  [ Random Forest to predict letter B]
    # Making the model
set.seed(456)
model_randomforest <- randomForest(as.factor(isB) ~.-letter, 
                                   data = letters_orig.train)
    # Predicting the outcomes
pred_rf <- predict(model_randomforest, newdata = letters_orig.test)
    # Confusion Matrix
table(pred_rf, letters_orig.test$isB)
  # pred_rf   FALSE TRUE
  #     FALSE   781   21
  #     TRUE      7  282

randomforest_acc <- (781 + 282)/ (781+281+7+22)
randomforest_acc
    # This value is 0.9743355

# Part B

# Part B.i    [ Baseline Model ]
set.seed(456)
table(letters_orig.train$letter)
      #  A   B   P   R 
      # 546 463 520 496
  # Thus, the prediction for every observation is 'A'
table(letters_orig.test$letter)
      #    A   B   P   R 
      #  243 303 283 262 
baseline_acc_part_b <- 243 / (243 + 303 + 283 + 262)
baseline_acc_part_b
      # The value of accuracy is 0.2227 for the baseline model

# Part B.ii   [ Linear Discriminant Analysis ]
set.seed(456)
model_lda <- lda(letter~.-isB, 
                 data = letters_orig.train)
pred_lda <- predict(model_lda, newdata = letters_orig.test)
pred_lda_class <- pred_lda$class
pred_lda_probs <- pred_lda$posterior

tab <- table(letters_orig.test$letter, pred_lda_class)
tab
#         pred_lda_class
#           A     B     P     R
#     A   227     5     1    10
#     B     0   272     0    31
#     P     0     6   273     4
#     R     0    31     1   230
lda_accuracy_part_b <- sum(diag(tab))/ sum(tab)
lda_accuracy_part_b
    # The value of model accuracy for this case of LDA is 0.9184


# Part B.iii    [ CART Model]
set.seed(456)
cpVals_part_b = data.frame(cp = seq(0, .002, by=.0001))
model_cart_part_b <- train(letter ~ . - isB, 
                           data = letters_orig.train, 
                           method = "rpart", 
                           tuneGrid = cpVals_part_b,
                           trControl = trainControl(method = "cv", number=10),
                           metric = "Accuracy" )
model_cart_part_b$results
model_cart_part_b

# Now, for this model, we predict the test set values and calculate test set accuracy
set.seed(456)
pred_cart_part_b <- predict(model_cart_part_b$finalModel, newdata = letters_orig.test, type = "prob")
pred_cart_part_b
pred_cart_part_b.extended <- mutate(as.data.frame(pred_cart_part_b), Pred = ifelse(A>=0.5, 'A', ifelse(B>=0.5, "B", ifelse(P>=0.5, "P" , "R"))))

tab_cart_part_b <- table(letters_orig.test$letter, pred_cart_part_b.extended$Pred)
tab_cart_part_b
    #         A     B     P     R
    #   A   230     5     0     8
    #   B    10   253     6    34
    #   P     2    13   265     3
    #   R    10    27     1   224

cart_acc_part_b <- (sum(diag(tab_cart_part_b)))/ sum(tab_cart_part_b)
cart_acc_part_b
    # The accuracy for this model is 0.89092

# Part B.iv     [ Bagging of CART Models]
set.seed(456)
ncol(letters_orig.test)
    # The number of columns are 18
model_bagging <- randomForest(letter ~ . - isB, 
                              data = letters_orig.train, 
                              mtry = 16 )
model_bagging
    # Predicting the outcomes
set.seed(456)
pred_bagging <- predict(model_bagging, newdata = letters_orig.test)
pred_bagging
    # Confusion Matrix
bagging_tab <- table(pred_bagging, letters_orig.test$letter)
bagging_tab
#  pred_bagging   
#        A   B   P   R
#    A 236   2   0   1
#    B   4 280   2  19
#    P   1   2 278   2
#    R   2  19   3 240

bagging_acc <- sum(diag(bagging_tab))/sum(bagging_tab)
bagging_acc
    # The value for bagging accuracy is 0.9477

# Part B.v    [ Random forest with cross validation ]
set.seed(456)
mtry_part_b = data.frame(mtry = seq(1, 16, by=1))
model_randomforests_part_b <- train(letter ~ . - isB, 
                                    data = letters_orig.train, 
                                    method = "rf", 
                                    tuneGrid = mtry_part_b,
                                    trControl = trainControl(method = "cv", number=10),
                                    metric = "Accuracy" )

model_randomforests_part_b$results
# This table shows the best value accuracy at mtry = 6

final_rf_part_b <- model_randomforests_part_b$finalModel
final_rf_part_b

# Making the predictions on the test set
set.seed(456)
pred_rf_part_b <- predict(final_rf_part_b, letters_orig.test)
pred_rf_part_b

tab_rf_part_b <- table(pred_rf_part_b, letters_orig.test$letter) 
tab_rf_part_b

#  pred_rf_part_b   A     B     P     R
#            A    238     1     0     0
#            B      2   286     2    10
#            P      1     0   279     2
#            R      2    16     2   250

#Now, calculating the accuracy of the model, we get:
rf_acc_part_b <- sum(diag(tab_rf_part_b))/sum(tab_rf_part_b)
rf_acc_part_b
# The value of accuracy for the best model using cross validation 
# on Random Forests is 0.9651

# Part B.vi   [ Boosting ]
set.seed(456)
model_boosting <- gbm(letter ~ . -isB,
                      data = letters_orig.train,
                      distribution = "multinomial",
                      n.trees = 3300,
                      interaction.depth = 10)

# Making the Predictions on the boosting model
set.seed(456)
pred_boost <- predict(model_boosting, newdata = letters_orig.test, n.trees = 3300, type = "response")
pred_boost
# Converting the posterior probabilities into a prediction
final_pred_part_b = apply(pred_boost, 1, which.max)
final_pred_part_b = factor(final_pred_part_b, levels = c(1,2,3,4), labels = c("A", "B", "P", "R"))

# The Confusion Matrix for Boosting
tab_boosting_part_b <- table(final_pred_part_b, letters_orig.test$letter)
tab_boosting_part_b

#  final_pred_part_b    A     B     P     R
#                  A  240     0     0     0
#                  B    2   295     1     5
#                  P    0     0   277     0
#                  R    1     8     5   257

# Calculating the accuracy of the model, we get
boosting_acc <- sum(diag(tab_boosting_part_b))/sum(tab_boosting_part_b)
boosting_acc
# The value of prediction accuracy in this case is 0.979835