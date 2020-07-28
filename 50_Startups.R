# Build a Neural Network model for 50_startups data to predict profit 

startups_data <- read.csv(file.choose())
str(startups_data)
summary(startups_data)

## lets convert state in numeric

startups_data$State <- as.numeric(startups_data$State,c("New York"="0","California"="1",
                                                                "Florida"="2"))

str(startups_data)

library(moments)

hist(startups_data$Profit)

plot(startups_data$R.D.Spend,startups_data$Profit) ## there is stromg positive relationship
plot(startups_data$Administration,startups_data$Profit) ## there is relationship but not strong
plot(startups_data$Marketing.Spend,startups_data$Profit) ## there is slight positive relationship

## lets check corelation coefficients

pairs(startups_data)

cor(startups_data)

#                R.D.Spend  Administration   Marketing.Spend State    Profit
# R.D.Spend       1.0000000     0.24195525      0.72424813 0.10468511 0.9729005
# Administration  0.2419552     1.00000000     -0.03215388 0.01184720 0.2007166
# Marketing.Spend 0.7242481    -0.03215388      1.00000000 0.07766961 0.7477657
# State           0.1046851     0.01184720      0.07766961 1.00000000 0.1017963
# Profit          0.9729005     0.20071657      0.74776572 0.10179631 1.0000000

## lets normalise data to get data into one range

startups_normalise <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

data_startups <- as.data.frame(lapply(startups_data,startups_normalise))


## lets split data into train and test
library(caTools)

spliting_data <- sample.split(data_startups$Profit,SplitRatio = 0.60)

startups_train <- subset(data_startups,spliting_data==TRUE)

startups_test <- subset(data_startups,spliting_data==FALSE)

## now will build our firsdt model on train data set using ANN

library(neuralnet)

attach(startups_train)

startups_model1 <- neuralnet(Profit ~.,data = startups_train)
plot(startups_model1) ## error is 0.047

startups_pred1 <- compute(startups_model1,startups_train[1:4])
cor(startups_pred1$net.result,startups_train$Profit) ## 0.9711952

## lets check accuracy on test dataset

startups_pred <- compute(startups_model1,startups_test[1:4])
cor(startups_pred$net.result,startups_test$Profit) ## 0.9559429

## will try to increase accuracy of model

startups_model2 <- neuralnet(Profit ~ .,data = startups_train,hidden = c(1,2,3))
plot(startups_model2) ## error is 0.05

startups_pred2 <- compute(startups_model2,startups_train[1:4])
cor(startups_pred2$net.result,startups_train$Profit) ## 0.9645645

## lets check accuracy on test dataset

startups_pred_test <- compute(startups_model2,startups_test[1:4])
cor(startups_pred_test$net.result,startups_test$Profit) ## 0.9664119

## again will some other values for hidden

startups_model3 <- neuralnet(Profit ~ .,data = startups_train,hidden = c(7,7))
plot(startups_model3) ## error 0.03

startups_pred3 <- compute(startups_model3,startups_train[1:4])
cor(startups_pred3$net.result,startups_train$Profit) ## 0.9796554

## lets check accuracy on test dataset

startups_pred_test1 <- compute(startups_model3,startups_test[1:4])
cor(startups_pred_test1$net.result,startups_test$Profit) ## 0.9760449

## below are accuries we got 

# startups_model1  0.9711952
# startups_model2  0.9645645
# startups_model3  0.9796554

## higher accuracy we got for startups_model3 so will accept this model
