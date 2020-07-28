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

plot(startups_data$R.D.Spend,startups_data$Profit) 
plot(startups_data$Administration,startups_data$Profit)
plot(startups_data$Marketing.Spend,startups_data$Profit) 

## lets check corelation coefficients

pairs(startups_data)

cor(startups_data)

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
plot(startups_model1) 

startups_pred1 <- compute(startups_model1,startups_train[1:4])
cor(startups_pred1$net.result,startups_train$Profit) 

## lets check accuracy on test dataset

startups_pred <- compute(startups_model1,startups_test[1:4])
cor(startups_pred$net.result,startups_test$Profit) 

## will try to increase accuracy of model

startups_model2 <- neuralnet(Profit ~ .,data = startups_train,hidden = c(1,2,3))
plot(startups_model2) 

startups_pred2 <- compute(startups_model2,startups_train[1:4])
cor(startups_pred2$net.result,startups_train$Profit) 

## lets check accuracy on test dataset

startups_pred_test <- compute(startups_model2,startups_test[1:4])
cor(startups_pred_test$net.result,startups_test$Profit) 

## again will some other values for hidden

startups_model3 <- neuralnet(Profit ~ .,data = startups_train,hidden = c(7,7))
plot(startups_model3) 

startups_pred3 <- compute(startups_model3,startups_train[1:4])
cor(startups_pred3$net.result,startups_train$Profit) 

## lets check accuracy on test dataset

startups_pred_test1 <- compute(startups_model3,startups_test[1:4])
cor(startups_pred_test1$net.result,startups_test$Profit) 

