setwd('E:\\Work\\AV\\BigMartSalesIII\\');

## Reading the training data set
data_train = read.csv("train.csv");
data_train$is_train = 1;
data_test <- read.csv("test.csv")
data_test$is_train = 0 ;
names(data_train)
data_test$Item_Outlet_Sales = mean(data_train$Item_Outlet_Sales);
data_train <- rbind(data_train , data_test)

table(data_train$is_train)
## General data checks
summary(data_train);
str(data_train)

summary(data_train$Item_Weight)
data_train$Item_Weight[is.na(data_train$Item_Weight)] = mean(data_train$Item_Weight, na.rm=TRUE)

str(data_train)
sum(is.na(data_train))

library(caret)
# dummify the data
dmy <- dummyVars(" ~ .", data = data_train)
trsf <- data.frame(predict(dmy, newdata = data_train))
print(trsf)

head(trsf)
data_train <- trsf

data_test <- data_train[data_train$is_train == 0,]
data_train <- data_train[data_train$is_train == 1,]


### SVM algorithm
set.seed(400)
ctrl <- trainControl(method="repeatedcv",repeats = 3)
###  with C you can adjust how hard or soft your large margin classification should be
grid <- expand.grid(C = seq(6,10,by=0.5))

svmModel <- train(Item_Outlet_Sales ~ ., data=data_train ,
                  method = "svmLinear", 
                trControl = ctrl, 
                tuneGrid = grid,
              ##preProcess = c("center","scale"),
                tuneLength = 25)
svmModel


#Use plots to see optimal number of clusters:
#Plotting yields Number of Neighbours Vs accuracy (based on repeated cross validation)
plot(svmModel)
## Look at model summary
pred <- predict(svmModel , newdata= data_test)

summary(pred)

##pred <- ifelse(model == 1 , 'N' , 'Y')
data_test1 <- read.csv("test.csv")
final_submission <- c(data.frame(data_test1$Item_Identifier, data_test1$Outlet_Identifier),data.frame(pred))

head(read.csv('SampleSubmission.csv',header=T))
names(final_submission) <- names(read.csv('SampleSubmission.csv',header=T));
soln="svmLinearC10.0";
write.csv(final_submission,file=paste(soln,".csv"),row.names = FALSE);

