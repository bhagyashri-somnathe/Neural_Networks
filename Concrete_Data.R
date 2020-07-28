# Prepare a model for strength of concrete data using Neural Networks

concrete_data <- read.csv(file.choose())
str(concrete_data)
summary(concrete_data)

## here data is available in diff ranges ,so we wil normalize data 

normalise_data <- function(x){
  return ((x-min(x)) / (max(x)-min(x)))
}

normalise_concrete <- as.data.frame(lapply(concrete_data,FUN = normalise_data))
summary(normalise_concrete) ## now we got data in one range

## now we will split data into train and test

library(caTools)

spliting_data <- sample.split(concrete_data$strength,SplitRatio = 0.70)

## here we have splited data into 7030 ratio and it is random sampling we have done

concrete_train <- subset(normalise_concrete,spliting_data == TRUE)

## here we are creating train data by using subset function and wherever data is matching with True it will get transfred totrain data

concrete_test <- subset(normalise_concrete,spliting_data ==  FALSE)

## here we have completed our EDA process and now we can build our model 

## lets build our model using ANN 

install.packages("neuralnet")
install.packages("nnet")

library(neuralnet) ## this library is used for regression
library(nnet)  ## this library is used for classification

## here our output value is in numerical value so we will use neuralnet library to build our model

attach(concrete_train)

concrete_model <- neuralnet(strength ~ .,data = concrete_train)

plot(concrete_model) 
## lets check performance of model on train data

train_result <- compute(concrete_model,concrete_train[1:8])

cor(train_result$net.result,concrete_train$strength) 

## lets check performance of model on test data

test_result <- compute(concrete_model,concrete_test[1:8])

cor(test_result$net.result,concrete_test$strength) 

## lets try to improve accuracy of model

concrete_model1 <- neuralnet(strength ~ ., data = concrete_train,hidden = c(2,5))
## here we have given hiddeen parameter to increase accuracy i.e. in 2nd layer we want 5 more nodes

plot(concrete_model1) 

## now lets check performance on train data

concrete_model1_pred <- compute(concrete_model1,concrete_train[1:8])

cor(concrete_model1_pred$net.result,concrete_train$strength) 

## now lets check performance on test data

concrete_model1_test <- compute(concrete_model1,concrete_test[1:8])

cor(concrete_model1_test$net.result,concrete_test$strength)

## will try to increase the accuracy

concrete_model2 <- neuralnet(strength ~ .,data = concrete_train,hidden = c(5,5,6))
plot(concrete_model2)

concrete_model2_pred <- compute(concrete_model2,concrete_train[1:8])

cor(concrete_model2_pred$net.result,concrete_train$strength)

plot(concrete_model2_pred$net.result,concrete_train$strength)


## lets apply it on test data

concrete_model2_test <- compute(concrete_model2,concrete_test[1:8])

cor(concrete_model2_test$net.result,concrete_test$strength)

