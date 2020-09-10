# Homeowork Assignment 1 Question 3 
# Part a

library(dplyr)
library(ggplot2)
library(GGally)
library(car) 

#loading in the data
wrangler_orig <- read.csv("Wrangler242-Fall2019_gasPrice.csv")
View(wrangler_orig)

#Search for any corelation in the given data
ggscatmat(wrangler_orig, columns = 2:8, alpha = 0.8)
# We observe a strong correlation between 

#Splitting into training and testing set
training_set <- filter(wrangler_orig, Year<= 2015)
testing_set <- filter(wrangler_orig, Year >= 2016)

model1 <- lm(WranglerSales ~ Unemployment + WranglerQueries + CPI.Energy + CPI.All, data = training_set)
summary(model1)
vif(model1)
# model 1 has a very high VIF for CPI all and Unemployment
# removing CPI all

model2 <- lm(WranglerSales ~ Unemployment + WranglerQueries + CPI.Energy, data = training_set)
summary(model2)
vif(model2)
# the VIF for all the variables are acceptable, however the p value for Unemployment is not

model3 <- lm(WranglerSales ~ WranglerQueries + CPI.Energy, data = training_set)
summary(model3)
vif(model3)
# the VIF for both variables are acceptabel, however the p value for CPI.Energy

model4 <- lm(WranglerSales ~ WranglerQueries, data = training_set)
summary(model4)
vif(model4)
# the p value for Wrangler Queries are acceptable,
# the value of R squared has not changed much after removing
# the other variables, so my final model only will use Wrangler Queries
# to model the linear regression 

#testing model 1 for its OSR Squared
SalesPrediction1 <- predict(model1, newdata=testing_set)

SSE1 = sum((testing_set$WranglerSales - SalesPrediction1)^2)
SST1 = sum((testing_set$WranglerSales - mean(training_set$WranglerSales))^2)
OSRsquared1 = 1 - SSE1/SST1
#It is 0.45, not very helpful

#testing model 2 for its OSR Squared
SalesPrediction2 <- predict(model2, newdata=testing_set)

SSE2 = sum((testing_set$WranglerSales - SalesPrediction2)^2)
SST2 = sum((testing_set$WranglerSales - mean(training_set$WranglerSales))^2)
OSRsquared2 = 1 - SSE2/SST2
#It is 0.57, improved but not by much

#testing model 3 for its OSR Squared
SalesPrediction3 <- predict(model3, newdata=testing_set)

SSE3 = sum((testing_set$WranglerSales - SalesPrediction3)^2)
SST3 = sum((testing_set$WranglerSales - mean(training_set$WranglerSales))^2)
OSRsquared3 = 1 - SSE3/SST3
#It is 0.57, no improvement, not useful

#testing model 4 for its OSR Squared
SalesPrediction4 <- predict(model4, newdata=testing_set)

SSE4 = sum((testing_set$WranglerSales - SalesPrediction4)^2)
SST4 = sum((testing_set$WranglerSales - mean(training_set$WranglerSales))^2)
OSRsquared4 = 1 - SSE4/SST4
#It is 0.53, R squared has become worse




# Question 3 Part b (Considering Seasonality)
model5 <- lm(WranglerSales ~ Unemployment + WranglerQueries + CPI.Energy + CPI.All + MonthFactor, data = training_set)
summary(model5)
vif(model5)


# Question 3 Part c (Building a better model)
model6 <- lm(WranglerSales ~ WranglerQueries + MonthFactor, data = training_set)
summary(model6)
vif(model6)

#testing model 6 for its OSR Squared
SalesPrediction6 <- predict(model6, newdata=testing_set)

SSE6 = sum((testing_set$WranglerSales - SalesPrediction6)^2)
SST6 = sum((testing_set$WranglerSales - mean(training_set$WranglerSales))^2)
OSRsquared6 = 1 - SSE6/SST6
#It is 0.6499, R squared has become worse



# Question 3 part d (adding oil price data)
model7 <- lm(WranglerSales ~ WranglerQueries + MonthFactor + GasolinePrice, data = training_set)
summary(model7)
vif(model7)

#testing model 7 for its OSR Squared
SalesPrediction7 <- predict(model7, newdata=testing_set)

SSE7 = sum((testing_set$WranglerSales - SalesPrediction7)^2)
SST7 = sum((testing_set$WranglerSales - mean(training_set$WranglerSales))^2)
OSRsquared7 = 1 - SSE7/SST7
#It is 0.6499, R squared has become worse




