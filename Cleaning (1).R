library(caret)
library(catboost)
library(readr)
library(qacEDA)
library(caret)
library(visreg)
library(MASS)
library(ggthemes)
library(pROC)
library(plotROC)

##---------------------------------------------##
##ANN                  ##
##---------------------------------------------##
# Data Cleaning
satisfaction <- read.csv("airline_satisfaction.csv",stringsAsFactors = F)
options(scipen=999) 
satisfaction <- satisfaction[,-c(1)]
satisfaction$Gender <- factor(satisfaction$Gender)
satisfaction$Customer.Type <- factor(satisfaction$Customer.Type)
satisfaction$Type.of.Travel <- factor(satisfaction$Type.of.Travel)
satisfaction$Class <- factor(satisfaction$Class)
satisfaction$satisfaction_v2 <- ifelse(satisfaction$satisfaction_v2=="satisfied","Yes","No")
satisfaction$satisfaction_v2 <- factor(satisfaction$satisfaction_v2)
satisfaction <- na.omit(satisfaction)

# First Split
set.seed(1234)
index <- createDataPartition(satisfaction$satisfaction_v2, 
                             p=0.1, list=FALSE)
small_set <- satisfaction[index,]

# Second Split
set.seed(1234)
index <- createDataPartition(small_set$satisfaction_v2, 
                             p=0.7, list=FALSE)
train <- small_set[index,]
test <- small_set[-index,]
tab(train,satisfaction_v2)
train.control <- trainControl(method="cv", number=10,
                              classProbs=TRUE,
                              summaryFunction=twoClassSummary)

# Model ANN
set.seed(1234)
model.ann <- train(satisfaction_v2 ~ ., 
                   data=train,
                   method="nnet",
                   tuneLength=5,  
                   size=9,
                   metric="ROC",
                   trControl=train.control,
                   preProcess=c("range"))
# Variable Importance
model.ann
varImp(model.ann)
plot(varImp(model.ann))
# Model Performance
pred <- predict(model.ann, test)
confusionMatrix(pred, test$satisfaction_v2, positive="Yes")
# ROC Curve
train$prob <- predict(model.ann, train, type="prob")[[2]]
train$pred <- factor(train$prob >.5,
                     levels = c(FALSE, TRUE),
                     labels = c("No", "Yes"))
ggplot(train, aes(d=satisfaction_v2, m=prob)) +
  geom_roc(labelround=2, n.cuts=15, labelsize=3) + 
  labs(title="ROC Plot") + 
  style_roc(major.breaks=seq(0, 1, .1),
            minor.breaks=seq(0, 1, .05),
            theme=theme_grey)
auc(train$satisfaction_v2, train$prob)

##---------------------------------------------##
##KNN                  ##
##---------------------------------------------##

# Data Cleaning
satisfaction <- read.csv("airline_satisfaction.csv",stringsAsFactors = F)
options(scipen=999) 
satisfaction <- satisfaction[,-c(1)]
satisfaction$Gender <- factor(satisfaction$Gender)
satisfaction$Customer.Type <- factor(satisfaction$Customer.Type)
satisfaction$Type.of.Travel <- factor(satisfaction$Type.of.Travel)
satisfaction$Class <- factor(satisfaction$Class)
satisfaction$satisfaction_v2 <- ifelse(satisfaction$satisfaction_v2=="satisfied","Yes","No")
satisfaction$satisfaction_v2 <- factor(satisfaction$satisfaction_v2)
satisfaction <- na.omit(satisfaction)

# Split
set.seed(1234)
index <- createDataPartition(satisfaction$satisfaction_v2, 
                             p=0.1, list=FALSE)
small_set <- satisfaction[index,]
set.seed(1234)
index <- createDataPartition(small_set$satisfaction_v2, 
                             p=0.7, list=FALSE)
train <- small_set[index,]
test <- small_set[-index,]
tab(train,satisfaction_v2)

# Model KNN
set.seed(1234)
model.knn <- train(satisfaction_v2 ~., 
                   data=train, 
                   preProcess=c("center", "scale"), 
                   method="knn",
                   tuneGrid=data.frame(k=seq(3,30, 3)),
                   trControl=trainControl(method="cv", 
                                          number=10),
                   metric="Accuracy")
# Variable Importance
model.knn
plot(model.knn)
varImp(model.knn) 
plot(varImp(model.knn))
# Model Performance
pred <- predict(model.knn, test)
confusionMatrix(pred, test$satisfaction_v2, positive="Yes")
# ROC Curve
knnpred <- predict(model.knn, test)
confusionMatrix(knnpred, test$satisfaction_v2)
train$prob <- predict(model.knn, train, type="prob")[[2]]
train$pred <- factor(train$prob >.5,
                     levels = c(FALSE, TRUE),
                     labels = c("No", "Yes"))
ggplot(train, aes(d=satisfaction_v2, m=prob)) +
  geom_roc(labelround=2, n.cuts=15, labelsize=3) + 
  labs(title="ROC Plot") + 
  style_roc(major.breaks=seq(0, 1, .1),
            minor.breaks=seq(0, 1, .05),
            theme=theme_grey)
auc(train$satisfaction_v2, train$prob)

##---------------------------------------------##
##XGB                  ##
##---------------------------------------------##

# Data Cleaning
satisfaction <- read.csv("airline_satisfaction.csv",stringsAsFactors = F)
options(scipen=999) 
satisfaction <- satisfaction[,-c(1)]
satisfaction$Gender <- factor(satisfaction$Gender)
satisfaction$Customer.Type <- factor(satisfaction$Customer.Type)
satisfaction$Type.of.Travel <- factor(satisfaction$Type.of.Travel)
satisfaction$Class <- factor(satisfaction$Class)
satisfaction$satisfaction_v2 <- ifelse(satisfaction$satisfaction_v2=="satisfied","Yes","No")
satisfaction$satisfaction_v2 <- factor(satisfaction$satisfaction_v2)
satisfaction <- na.omit(satisfaction)

# Split
set.seed(1234)
index <- createDataPartition(satisfaction$satisfaction_v2, 
                             p=0.1, list=FALSE)
small_set <- satisfaction[index,]
set.seed(1234)
index <- createDataPartition(small_set$satisfaction_v2, 
                             p=0.7, list=FALSE)
train <- small_set[index,]
test <- small_set[-index,]
tab(train,satisfaction_v2)

# Modle XGB
train.control <- trainControl(method = "cv", 
                              number = 5)
start <- Sys.time()
set.seed(1234)
model.xgb <- train(satisfaction_v2~., 
                   data = train, 
                   method = "xgbTree", 
                   tuneLength=3,
                   trControl = train.control)
end <- Sys.time()
end - start

# Variable Importance
model.xgb
varImp(model.xgb)
plot(varImp(model.xgb))
# Modle Perforamce
predict <- predict(model.xgb, test)
confusionMatrix(predict, test$satisfaction_v2)
# ROC Curve
train$prob <- predict(model.xgb, train, type="prob")[[2]]
train$pred <- factor(train$prob >.5,
                     levels = c(FALSE, TRUE),
                     labels = c("No", "Yes"))
ggplot(train, aes(d=satisfaction_v2, m=prob)) +
  geom_roc(labelround=2, n.cuts=15, labelsize=3) + 
  labs(title="ROC Plot") + 
  style_roc(major.breaks=seq(0, 1, .1),
            minor.breaks=seq(0, 1, .05),
            theme=theme_grey)
auc(train$satisfaction_v2, train$prob)


#---------------------------------------------##
## Grow Classification Tree - ROC              ##
##---------------------------------------------##
# Data Cleaning
satisfaction <- read.csv("airline_satisfaction.csv",stringsAsFactors = F)
options(scipen=999) 
satisfaction <- satisfaction[,-c(1)]
satisfaction$Gender <- factor(satisfaction$Gender)
satisfaction$Customer.Type <- factor(satisfaction$Customer.Type)
satisfaction$Type.of.Travel <- factor(satisfaction$Type.of.Travel)
satisfaction$Class <- factor(satisfaction$Class)
satisfaction$satisfaction_v2 <- ifelse(satisfaction$satisfaction_v2=="satisfied","Yes","No")
satisfaction$satisfaction_v2 <- factor(satisfaction$satisfaction_v2)
satisfaction <- na.omit(satisfaction)

# Split
set.seed(1234)
index <- createDataPartition(satisfaction$satisfaction_v2, 
                             p=0.1, list=FALSE)
small_set <- satisfaction[index,]
set.seed(1234)
index <- createDataPartition(small_set$satisfaction_v2, 
                             p=0.7, list=FALSE)
train <- small_set[index,]
test <- small_set[-index,]
tab(train,satisfaction_v2)

# Model Tree
trctrl <- trainControl(method="cv", number=10,
                       summaryFunction=twoClassSummary,
                       classProbs=TRUE)
set.seed(1234)
ctree_fit <- train(satisfaction_v2 ~., data = train, 
                   method = "rpart",
                   trControl=trctrl,
                   metric = "ROC",
                   tuneLength = 15)
ctree_fit

# Plot tree
library(rattle)
fancyRpartPlot(ctree_fit$finalModel, tweak=1.5)

# Plot tree 2
library(rpart.plot)
fancyRpartPlot(ctree_fit$finalModel)
rpart.plot(ctree_fit$finalModel)
result <- rpart.rules(ctree_fit$finalModel)
result

# Variable Importance
varImp(ctree_fit)
plot(varImp(ctree_fit))

# Modle Performance 
pred <- predict(ctree_fit, test)
confusionMatrix(pred, test$satisfaction_v2, positive="Yes")

# ROC Curve
train$prob <- predict(ctree_fit, train, type="prob")[[2]]
train$pred <- factor(train$prob >.5,
                     levels = c(FALSE, TRUE),
                     labels = c("No", "Yes"))
ggplot(train, aes(d=satisfaction_v2, m=prob)) +
  geom_roc(labelround=2, n.cuts=15, labelsize=3) + 
  labs(title="ROC Plot") + 
  style_roc(major.breaks=seq(0, 1, .1),
            minor.breaks=seq(0, 1, .05),
            theme=theme_grey)
auc(train$satisfaction_v2, train$prob)

