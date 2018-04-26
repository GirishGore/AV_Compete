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

data_test <- data_train[data_train$is_train == 0,]
data_train <- data_train[data_train$is_train == 1,]

str(data_train)

library(dplyr)
data_train_sales <- group_by(data_train, Item_Identifier ,Outlet_Identifier) %>% 
                    summarize(Item_Outlet_Sales=mean(Item_Outlet_Sales))

data_submit <- read.csv("SampleSubmission.csv")
merge(data_submit , data_train_sales , by = c("Item_Identifier","Outlet_Identifier"))
head(data_submit)
head(data_train_sales)
str(data_train_sales)
str(data_submit)
data_train_sales$Item_Identifier = as.character(data_train_sales$Item_Identifier)
data_train_sales$Outlet_Identifier = as.character(data_train_sales$Outlet_Identifier)
data_submit$Item_Identifier = as.character(data_submit$Item_Identifier)
data_submit$Outlet_Identifier = as.character(data_submit$Outlet_Identifier)
