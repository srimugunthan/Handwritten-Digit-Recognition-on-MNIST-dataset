library("pracma")
library('Matrix')
library("nnet")


#the globals in this file
  # X
  # Y
  # numClasses
  # num_features
  # initialtheta
  # optTheta




#TODO:
# add the gradient calculation part too
# how to add the gradient check algorithm




softmaxcost <- function(theta)
{
    
    
    shapedtheta <- matrix(theta, nrow = num_features, byrow = TRUE)
    thetaTx <- X %*% shapedtheta
    yh <- exp(thetaTx)
    denom <- rowSums(yh)
    p_y <- sweep(yh,MARGIN=1,denom,"/")
    lgpy <- log(p_y)

        
    
    indicator <- class.ind(Y)
    cost <- -sum(rowSums(indicator * lgpy))
    
    
#     cost2 <- 0
#     for(i in 1:nrow(Y))
#     {
#       cost2 <- cost2 + lgpy[i,Y[i]] 
#     }
#     cost2 <- -cost2


    #thetagrad = (-1/m) * (y - p) * t(data) + lambda * theta;      
    
    #print(paste(" ",cost, cost2))          
    return (cost)
}

softmaxTrain <- function( traindata, Ypred)
{
  X <<- as.matrix(traindata)
  X <<- cbind(rep(1,nrow(X)),X)
  Y <<- as.matrix(Ypred)
  
  numClasses <<- length(levels(as.factor(Y)))
  num_features <<- ncol(X)
  
  #initialtheta <<- 0.005 *matrix(rnorm((numClasses * num_features)), 1)
  initialtheta <<- rep(0,(numClasses * num_features))
  initialtheta <<- matrix(initialtheta, nrow = num_features, byrow = TRUE)
  
  
  softmaxcost(initialtheta)
  theta_optim <- optim(par=initialtheta,fn=softmaxcost,method = "BFGS",
                       control = list(maxit = 10,  trace = TRUE))
  #theta_optim <- optim(par=initialtheta,fn=softmaxcost)
  softmaxOptTheta <- theta_optim$par
  
  #softmaxModel = new
  optTheta <<- matrix(softmaxOptTheta, nrow = num_features, byrow = TRUE)
  
  #num_features <<- num_features;
  #numClasses <<- numClasses;
  
  #return (softmaxModel)
  
}


#softmaxPredict <- function(softmaxModel, testdata)
softmaxPredict <- function(testdata)
{
  #theta <- softmaxModel.optTheta$par
  theta <-optTheta
  
  tdata <- cbind(rep(1,nrow(testdata)),testdata)
  #predprobs <- vector(mode="numeric", length=nrow(testdata))
  predprobs <- matrix(0,  nrow(tdata),numClasses)
  thetaTx <- (as.matrix(tdata) %*% (optTheta))
  numermat <- exp(thetaTx)
  
  denom <- rowSums(numermat)
  predprobs <- sweep(numermat,MARGIN=1,denom,"/")
  
#   for (i in 1:nrow(testdata))
#   {
#     y <- tdata[i,]
#     denom <- sum(exp(as.matrix(y) %*% optTheta))
#     for(k in 1:numClasses)
#     {
#       numer <-  exp(as.matrix(y) %*% as.matrix(theta[,k])) 
#       prob <-numer/denom
#       predprobs[i,k] <- prob
#     }
#   }
  
  return (predprobs)
}


dummy_mysoftmaxcall <-  function()
{
 print("dummy soft max") 
}


###################standalone test it with seeds dataset ###############
# 
# setwd("/home/srimugunthan/TookitakiEx")
# seeds <-  read.csv("seeds_dataset.txt",   header=FALSE, sep="", as.is=TRUE)
# seedstrain<- sample(1:210,147)
# seedstest <- setdiff(1:210,seedstrain)
# traindf <- seeds[seedstrain,]
# testdf <-  seeds[seedstest,]
#  
# 
#  softmaxTrain(traindata=traindf[1:7],  Ypred= traindf$V8)
#   #mysoftmaxmodel <-softmaxTrain(traindata=traindf[1:7],  Ypred= traindf$V8)
# 
# ###
# 
# predprobs <- softmaxPredict(testdata = testdf[1:7])
# 
# 
# ##
# cum.probs <- t(apply(predprobs,1,cumsum))
# 
# # Draw random values
# vals <- runif(nrow(testdf))
# 
# # Join cumulative probabilities and random draws
# tmp <- cbind(cum.probs,vals)
# 
# # For each row, get choice index.
# k <- ncol(predprobs)
# predresults <- 1 + apply(tmp,1,function(x) length(which(x[1:k] < x[k+1])))
# 
# results <- table(predresults,testdf$V8)
# 
# indicatormat <- class.ind(traindf$V8)
# sink(file = "analysis-output.txt")
# cat("=====================================================\n")
# print(results)
# cat("=====================================================\n")
# sink(NULL)