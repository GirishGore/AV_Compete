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

data_train <- trsf

data_test <- data_train[data_train$is_train == 0,]
data_train <- data_train[data_train$is_train == 1,]

library(caret)
### GBM algorithm
fitControl <- trainControl(## 10-fold CV
  method = "cv",
  number = 10,
  ## repeated ten times
  ##repeats = 3
  );

gbmGrid <-  expand.grid(interaction.depth = c(7,11,17),
                        n.trees = 1000, #(1:30)*50
                        shrinkage = c(0.1,0.01),
                        n.minobsinnode = 5);

gbmFit <- train(Item_Outlet_Sales ~ ., data = data_train,
                 method = "gbm",
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = TRUE,
                 tuneGrid = gbmGrid)


#Use plots to see optimal number of clusters:
#Plotting yields Number of Neighbours Vs accuracy (based on repeated cross validation)
gbmFit
## Look at model summary
pred <- predict(gbmFit , newdata= data_test)

summary(pred)

data_test1 <- read.csv("test.csv")
final_submission <- c(data.frame(data_test1$Item_Identifier, data_test1$Outlet_Identifier),data.frame(pred))

head(read.csv('SampleSubmission.csv',header=T))
names(final_submission) <- names(read.csv('SampleSubmission.csv',header=T));
soln="gbm";
write.csv(final_submission,file=paste(soln,".csv"),row.names = FALSE);

