library("caret")
library("gains")
library("glmnet")
library("neuralnet")
library("nnet")
library("devtools")
library("pROC")
mainDir="/Users/sdhandap/Handwritten-Digit-Recognition-on-MNIST-dataset"
setwd(mainDir)

source("loadMNIST.R")
source("binomiallogistic.R")
source("mysoftmaxImplem.R")
source("mySupervisedNNET.R")


# glmnetmodel <- function(traindat, testdat)
# {
#   subDir="glmnetmodel"  
#   dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
#   setwd(file.path(mainDir, subDir))
#   
#   # http://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html
# 
# 
# 
#     cvfit=cv.glmnet(traindat$x, traindat$y, family="multinomial", type.multinomial = "grouped", parallel = TRUE)
#     pred <- predict(cvfit,  newx=traindat$x, s = "lambda.min", type = "class")
# 
#     sink(file = "analysis-output.txt",append=FALSE)
#     print(head(pred))
#     cat("========================GLM net model========================\n")
#     print(summary(cvfit))
#     
#     cat("---------------------------------------------------------------------\n")
#     #print(confusionMatrix(testset$predX1, testset$X1))
#     sink(NULL)
#   
#   setwd(mainDir)
# }


# neuralnetmodel <- function(traindf, testdf)
# {
#   subDir="neuralnetmodel"  
#   dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
#   setwd(file.path(mainDir, subDir))
# 
#   # http://stackoverflow.com/questions/10939591/predicting-class-for-new-data-using-neuralnet
#   # http://stackoverflow.com/questions/4891899/how-to-predict-new-cases-using-the-neuralnet-package
#   
#   # selecting particular columns
#   #columns <- c("x1","x2","x3","x4")
# 	#covariate <- subset(test, select = columns)
#   
#     n <- names(traindf)
#     f <- as.formula(paste("y ~", paste(n[!n %in% "y"], collapse = " + ")))
# 
#   
#     nnmodel <- neuralnet(
#     f ,
#     data=traindf, hidden=4,
#     linear.output=FALSE)
#   
#     pred = compute(nnmodel, traindf[,1:784])
# 
#     results <- data.frame(actual = traindf$y, predprob = pred$net.result, prediction =  round(pred$net.result))
#     #predictout <- round(pred$net.result)
#     #DataPred <- prediction(nnmodel, traindf[,1:784])
#   
#     sink(file = "analysis-output.txt",append=FALSE)
#     print((results))
#     cat("---------------------------------------------------------------------\n")
#     cat("------------Confusion matrix for train dataset------------------------\n")
#     print(confusionMatrix(results))
#     cat("---------------------------------------------------------------------\n")
#     #print(head(DataPred))
#     cat("========================Result matrix========================\n")
#     print(pred$result.matrix)
#     cat("========================Neural net model========================\n")
#     print(summary(nnmodel))
#   
#   
#     
#     cat("---------------------------------------------------------------------\n")
#     #print(confusionMatrix(testset$predX1, testset$X1))
#     sink(NULL)
#   
#     pdf("nnplot.pdf", width=7, height=7)
#   
#     plot(nnmodel)
#     dev.off()
#   
# 
# 
#     setwd(mainDir)
# }


onevsallmodel <- function(train,test)
{
  subDir="multinomialmodel"  
  dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
  setwd(file.path(mainDir, subDir))
  
#   train<-add_one_binary_var()
#   logit1 <- glm(train$y2 ~ ., data = train, family = "binomial")
#   testset$predX1prob <- predict(logit1, testset, type="response") 
#   testset$predX1 <- as.numeric(testset$predX1prob > 0.5)
#   
#   sink(file = "analysis-output.txt",append=FALSE)
#   cat("========================Logisticregression1========================\n")
#   print(summary(logit1))
#   
#   cat("---------------------------------------------------------------------\n")
#   print(confusionMatrix(testset$predX1, testset$X1))
#   sink(NULL)
  
  setwd(mainDir)
}


nnetmodel <- function(traindf, testdf)
{
  subDir="nnetmodel"  
  dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
  setwd(file.path(mainDir, subDir))
  
  
  ideal <- class.ind(traindf$y)
  
  start.time <- proc.time()
  nnetmodel = nnet(traindf[,1:784], ideal, size=10, softmax=TRUE,  MaxNWts=100000 )
  end.time <- proc.time()
  Training.time.taken <- end.time - start.time
  
  predstart.time <- proc.time()
  predresults <- predict(nnetmodel, traindf[,1:784], type="class")
  predend.time <- proc.time()
  Pred.time.taken <- predend.time - predstart.time
  
  
  results <- table(predresults,traindf$y)
  rocobj <- roc(predresults,traindf$y)
  trainauc <- auc(rocobj)
  
  sink(file = "analysis-output.txt",append=FALSE)
  print(nnetmodel)
  cat("-------------------------------Train set prediction results-------------\n")
  print((results))
  cat("--------------------------Time taken--------------------------------------\n")
  print("Training time in secs:")
  print(Training.time.taken)
  print("---")
  print("Prediction time in secs:")
  print(Pred.time.taken)
  print("---")
  
  
  cat("------------Confusion matrix for train dataset------------------------\n")
  print(confusionMatrix(results))
  cat("---------------------------------------------------------------------\n")
  cat("=====================ROC value================================\n")
  print(auc(rocobj))
  cat("---------------------------------------------------------------------\n")
  sink(NULL)
  
  ##########now to test set ##########
  
  
  predstart.time <- proc.time()
  predresults <- predict(nnetmodel, testdf[,1:784], type="class")
  predend.time <- proc.time()
  Pred.time.taken <- predend.time - predstart.time
  
  
  results <- table(predresults,testdf$y)
  rocobj <- roc(predresults,testdf$y)
  testauc <- auc(rocobj)
  sink(file = "analysis-output.txt",append=TRUE)
  cat("-------------------------------Test set prediction results-------------\n")
  
  
  print("Prediction time in secs:")
  print(Pred.time.taken)
  print("---")
  
  cat("------------Confusion matrix for test dataset------------------------\n")
  print(confusionMatrix(results))
  cat("---------------------------------------------------------------------\n")
  cat("=====================ROC value================================\n")
  print(auc(rocobj))
  cat("---------------------------------------------------------------------\n")
  sink(NULL)
  setwd(mainDir)
  return (c(trainauc, testauc))
}


multinomialmodel <- function(traindf, testdf)
{
  subDir="multinomialmodel"  
  dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
  setwd(file.path(mainDir, subDir))

  
  n <- names(traindf)
  f <- as.formula(paste("y ~", paste(n[!n %in% "y"], collapse = " + ")))
  

  
  start.time <- proc.time()
  multinommodel <- multinom(f, data = traindf, MaxNWts=100000 )
  end.time <- proc.time()
  Training.time.taken <- end.time - start.time
  
  



  
  pp <- fitted(multinommodel)

  #dses <- data.frame(ses = c("0", "1", "2"), write = mean(ml$write))
  predstart.time <- proc.time()
  predprobs <- predict(multinommodel, newdata = traindf, "probs")
  predend.time <- proc.time()
  Pred.time.taken <- predend.time - predstart.time
  
  cum.probs <- t(apply(predprobs,1,cumsum))
  
  # Draw random values
  vals <- runif(nrow(traindf))
  
  # Join cumulative probabilities and random draws
  tmp <- cbind(cum.probs,vals)
  
  # For each row, get choice index.
  k <- ncol(predprobs)
  predresults <- apply(tmp,1,function(x) length(which(x[1:k] < x[k+1])))
  
  results <- table(predresults,traindf$y)
  rocobj <- roc(predresults,traindf$y)
  trainauc <- auc(rocobj)
  sink(file = "analysis-output.txt")
  cat("-------------------------------Train set prediction results-------------\n")
  print((results))
  cat("--------------------------Time taken--------------------------------------\n")
  print("Training time in secs:")
  print(Training.time.taken)
  print("---")
  print("Prediction time in secs:")
  print(Pred.time.taken)
  print("---")
  
  cat("------------Confusion matrix for train dataset------------------------\n")
  print(confusionMatrix(results))
  cat("---------------------------------------------------------------------\n")
  cat("=====================ROC value================================\n")
  print(auc(rocobj))
  cat("---------------------------------------------------------------------\n")
  sink(NULL)
  
  ##########now to test set ##########
    
  
  
  predstart.time <- proc.time()
  #dses <- data.frame(ses = c("0", "1", "2"), write = mean(ml$write))
  predprobs <- predict(multinommodel, newdata = testdf, "probs")
  predend.time <- proc.time()
  Pred.time.taken <- predend.time - predstart.time
  
  cum.probs <- t(apply(predprobs,1,cumsum))
  
  # Draw random values
  vals <- runif(nrow(testdf))
  
  # Join cumulative probabilities and random draws
  tmp <- cbind(cum.probs,vals)
  
  # For each row, get choice index.
  k <- ncol(predprobs)
  predresults <-  apply(tmp,1,function(x) length(which(x[1:k] < x[k+1])))
  
  
  results <- table(predresults,testdf$y)
  rocobj <- roc(predresults,testdf$y)
  testauc <- auc(rocobj)
  sink(file = "analysis-output.txt",append=TRUE)
  cat("-------------------------------Test set prediction results-------------\n")
  print((results))
  
  print("Prediction time in secs:")
  print(Pred.time.taken)
  print("---")
  
  cat("------------Confusion matrix for test dataset------------------------\n")
  print(confusionMatrix(results))
  cat("---------------------------------------------------------------------\n")
  cat("=====================ROC value================================\n")
  print(auc(rocobj))
  cat("---------------------------------------------------------------------\n")
  sink(NULL)
  
  
  setwd(mainDir)
  return (c(trainauc, testauc))
}






mysimple_logistic <- function(traindf, testdf)
{
  subDir="mysimplelogistic"  
  dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
  setwd(file.path(mainDir, subDir))

  trainX <- as.matrix(traindf[,1:784])
  testX <- as.matrix(traindf[, 1:784])
  predresults <- mysimplelogistic(trainX, traindf$y,testX)
  
  #dses <- data.frame(ses = c("0", "1", "2"), write = mean(ml$write))
  #predresults <- mylogisticpredict(bilogisticmodel, newdata = traindf)
  
  results <- table(predresults,traindf$y)
  
  
  sink(file = "analysis-output.txt",append=FALSE)
  cat("========================MySimple Logistic model========================\n")
  print(results)
  cat("------------Confusion matrix for train dataset------------------------\n")
  print(confusionMatrix(results))
  cat("---------------------------------------------------------------------\n")
  cat("---------------------------------------------------------------------\n")
  sink(NULL)
  
  setwd(mainDir)
}


mysoftmax <- function(traindf, testdf)
{
  subDir="mysoftmax"  
  dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
  setwd(file.path(mainDir, subDir))
  
  
  start.time <- proc.time()
  
  softmaxTrain(traindata=traindf[,1:784],  Ypred= traindf$y)
  
  end.time <- proc.time()
  Training.time.taken <- end.time - start.time
  
  
  #mysoftmaxmodel <-softmaxTrain(traindata=traindf[1:7],  Ypred= traindf$V8)
  #predprobs <- predict(mysoftmaxmodel, testdata = traindf[1:7])
  
  
  predstart.time <- proc.time()
  #dses <- data.frame(ses = c("0", "1", "2"), write = mean(ml$write))
  predprobs <- softmaxPredict(testdata = traindf[,1:784])
  predend.time <- proc.time()
  Pred.time.taken <- predend.time - predstart.time
  
  # 
  cum.probs <- t(apply(predprobs,1,cumsum))
  
  # Draw random values
  vals <- runif(nrow(traindf))
  
  # Join cumulative probabilities and random draws
  tmp <- cbind(cum.probs,vals)
  
  # For each row, get choice index.
  k <- ncol(predprobs)
  predresults <-  apply(tmp,1,function(x) length(which(x[1:k] < x[k+1])))
  
  results <- table(predresults,traindf$y)
  rocobj <- roc(predresults,traindf$y)
  
  sink(file = "analysis-output.txt")
  cat("-------------------------------Train set prediction results-------------\n")
  print((results))
  cat("--------------------------Time taken--------------------------------------\n")
  print("Training time in secs:")
  print(Training.time.taken)
  print("---")
  print("Prediction time in secs:")
  print(Pred.time.taken)
  print("---")
  
  cat("------------Confusion matrix for train dataset------------------------\n")
  print(confusionMatrix(results))
  cat("---------------------------------------------------------------------\n")
  cat("=====================ROC value================================\n")
  print(auc(rocobj))
  cat("---------------------------------------------------------------------\n")
  sink(NULL)
  
  ################now to test set ############
  
  
  
  predstart.time <- proc.time()
  #dses <- data.frame(ses = c("0", "1", "2"), write = mean(ml$write))
  predprobs <- softmaxPredict(testdata = testdf[,1:784])
  predend.time <- proc.time()
  Pred.time.taken <- predend.time - predstart.time
  
  
  
  
  # 
  cum.probs <- t(apply(predprobs,1,cumsum))
  
  # Draw random values
  vals <- runif(nrow(testdf))
  
  # Join cumulative probabilities and random draws
  tmp <- cbind(cum.probs,vals)
  
  # For each row, get choice index.
  k <- ncol(predprobs)
  predresults <-  apply(tmp,1,function(x) length(which(x[1:k] < x[k+1])))
  
  results <- table(predresults,testdf$y)
  
  rocobj <- roc(predresults,testdf$y)
  
  sink(file = "analysis-output.txt",append=TRUE)
  cat("-------------------------------Test set prediction results-------------\n")
  print((results))
  
  print("Prediction time in secs:")
  print(Pred.time.taken)
  print("---")
  
  cat("------------Confusion matrix for test dataset------------------------\n")
  print(confusionMatrix(results))
  cat("---------------------------------------------------------------------\n")
  cat("=====================ROC value================================\n")
  print(auc(rocobj))
  cat("---------------------------------------------------------------------\n")
  sink(NULL)
  
  
  setwd(mainDir)
}


myneuralnet <- function(traindf, testdf)
{
  subDir="myNnet"  
  dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
  setwd(file.path(mainDir, subDir))
  
  start.time <- proc.time()
  
  myNnetTrain(traindata=traindf[,1:784],  Ypred= traindf$y)
  
  end.time <- proc.time()
  Training.time.taken <- end.time - start.time
  
  
  
  predstart.time <- proc.time()
  #dses <- data.frame(ses = c("0", "1", "2"), write = mean(ml$write))
  predprobs <- myNnetPredict(testdata = traindf[,1:784])
  predend.time <- proc.time()
  Pred.time.taken <- predend.time - predstart.time
  
  
  
  
  
  cum.probs <- t(apply(predprobs,1,cumsum))
  
  # Draw random values
  vals <- runif(nrow(traindf))
  
  # Join cumulative probabilities and random draws
  tmp <- cbind(cum.probs,vals)
  
  # For each row, get choice index.
  k <- ncol(predprobs)
  predresults <-  apply(tmp,1,function(x) length(which(x[1:k] < x[k+1])))
  
  results <- table(predresults,traindf$y)
  rocobj <- roc(predresults,traindf$y)
  
  sink(file = "analysis-output.txt")
  cat("-------------------------------Train set prediction results-------------\n")
  print((results))
  cat("--------------------------Time taken--------------------------------------\n")
  print("Training time in secs:")
  print(Training.time.taken)
  print("---")
  print("Prediction time in secs:")
  print(Pred.time.taken)
  print("---")
  
  cat("------------Confusion matrix for train dataset------------------------\n")
  print(confusionMatrix(results))
  cat("---------------------------------------------------------------------\n")
  cat("=====================ROC value================================\n")
  print(auc(rocobj))
  cat("---------------------------------------------------------------------\n")
  sink(NULL)
  
  
  
  ################## now to test set ############
  
  
  predstart.time <- proc.time()
  #dses <- data.frame(ses = c("0", "1", "2"), write = mean(ml$write))
  predprobs <- myNnetPredict(testdata = testdf[,1:784])
  predend.time <- proc.time()
  Pred.time.taken <- predend.time - predstart.time
  
  
  
  # 
  cum.probs <- t(apply(predprobs,1,cumsum))
  
  # Draw random values
  vals <- runif(nrow(testdf))
  
  # Join cumulative probabilities and random draws
  tmp <- cbind(cum.probs,vals)
  
  # For each row, get choice index.
  k <- ncol(predprobs)
  predresults <-  apply(tmp,1,function(x) length(which(x[1:k] < x[k+1])))
  
  results <- table(predresults,testdf$y)
  rocobj <- roc(predresults,testdf$y)
  
  sink(file = "analysis-output.txt",append=TRUE)
  cat("-------------------------------Test set prediction results-------------\n")
  print((results))
  
  print("Prediction time in secs:")
  print(Pred.time.taken)
  print("---")
  
  cat("------------Confusion matrix for test dataset------------------------\n")
  print(confusionMatrix(results))
  cat("---------------------------------------------------------------------\n")
  cat("=====================ROC value================================\n")
  print(auc(rocobj))
  cat("---------------------------------------------------------------------\n")
  sink(NULL)
  
  
  setwd(mainDir)
  
}


glmlogistic <- function(traindf, testdf)
{
  subDir="glmlogistic"  
  dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
  setwd(file.path(mainDir, subDir))
  
  n <- names(traindf)
  f <- as.formula(paste("y ~", paste(n[!n %in% "y"], collapse = " + ")))
  
  logit <- glm(f, data = traindf, family = "binomial")
  
  
  
  predprob <- predict(logit, traindf[,1:784], type="response") 
  predresults <- as.numeric(predprob > 0.5)
  
    #dses <- data.frame(ses = c("0", "1", "2"), write = mean(ml$write))
  #predresults <- mylogisticpredict(bilogisticmodel, newdata = traindf)
  
  results <- table(predresults,traindf$y)
  
  
  sink(file = "analysis-output.txt",append=FALSE)
  cat("========================GLM Logistic model========================\n")
  print(results)
  cat("---------------------------------------------------------------------\n")
  sink(NULL)
  
  setwd(mainDir)
}

Multinomial_learning_curve <- function(traindf, testdf)
{
  trainlens <- c(4000,  8000,   12000,   16000,   18000)
  for (i in 1:length(trainlens)) 
  {
    
    subDir=paste("Mlcrun",i,sep="")
    
    dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
    
    setwd(file.path(mainDir, subDir))
    mainDir <<- file.path(mainDir, subDir)
    
    print (trainlens[i])
    subsettraindf <- traindf[sample(1:nrow(traindf),trainlens[i]),]
    
    rv <- multinomialmodel(subsettraindf,testdf)
    
    mainDir <<- "/home/srimugunthan/TookitakiEx"  
    setwd(file.path(mainDir))
    sink(file = "Mlcurve-result.txt",append=TRUE)
    print(c(trainlens[i] , rv))
    sink(NULL)
  }
  
}



NNET_learning_curve <- function(traindf, testdf)
{
  trainlens <- c(4000,  8000,   12000,   16000,   18000,20000)
  for (i in 1:length(trainlens)) 
  {
    
    subDir=paste("Nlcrun",i,sep="")
    
    dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
    
    setwd(file.path(mainDir, subDir))
    mainDir <<- file.path(mainDir, subDir)
    
    print (trainlens[i])
    subsettraindf <- traindf[sample(1:nrow(traindf),trainlens[i]),]
    
    rv <- nnetmodel(subsettraindf,testdf)
    
    mainDir <<- "/home/srimugunthan/TookitakiEx"  
    setwd(file.path(mainDir))
    sink(file = "Nlcurve-result.txt",append=TRUE)
    print(c(trainlens[i] , rv))
    sink(NULL)
  }
  
}


totlstart.time <- proc.time()


traindat <- load_image_file('./train-images-idx3-ubyte')
testdat  <- load_image_file('./t10k-images-idx3-ubyte')

traindat$y <- load_label_file('./train-labels-idx1-ubyte')
testdat$y <- load_label_file('./t10k-labels-idx1-ubyte')



 traindf <- as.data.frame(traindat$x)
 traindf$y <- traindat$y
 
 testdf <- as.data.frame(testdat$x)
 testdf$y <- testdat$y


# #show_digit(train$x[5,]) 
rm(traindat)
rm(testdat)
gc()
# 
#lmtraindf <- subset(traindf,((traindf$y == 0) | (traindf$y == 1)))
#lmtestdf <- subset(testdf,((testdf$y == 0) | (testdf$y == 1)))

#samplestrain <- sample(1:nrow(traindf),20000)
#remsamples <- setdiff(1:nrow(traindf),samplestrain )
#smalltraindf <- traindf[samplestrain,]
smalltraindf <- traindf[sample(1:nrow(traindf),2000),]

#remtraindf <-   traindf[remsamples,]

#smalltraindf <- traindf[1:20000,]
#smalltraindf <- head(traindf,10000)
rm(traindf)
#rm(testdf)
gc()
#devtools::install_github("krlmlr/ulimit")
#ulimit::memory_limit(2000)

#mysimple_logistic(lmtraindf,lmtestdf)
#glmlogistic(lmtraindf,lmtestdf)

#multinomialmodel(smalltraindf,testdf)
nnetmodel(smalltraindf,testdf)
#myneuralnet(smalltraindf,testdf)
#mysoftmax(smalltraindf,testdf)

#Multinomial_learning_curve(smalltraindf,testdf)
#NNET_learning_curve(smalltraindf,testdf)

sink(type="output")
totlend.time <- proc.time()
Total.time.taken <- totlend.time - totlstart.time
print("Total time taken in  secs:")
print(Total.time.taken)
print(" ----------")