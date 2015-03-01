

#Sigmoid function
sigmoid <- function(z)
{
  g <- 1/(1+exp(-z))
  return(g)
}



#Cost Function
cost <- function(theta)
{
  m <- nrow(X)
  g <- sigmoid(X%*%theta)
  J <- (1/m)*sum((-Y*log(g)) - ((1-Y)*log(1-g)))
  
  return(J)
}



mysimplelogistic <- function(traindata, Ypred,testdata)
{

  X <- as.matrix(traindata)
  #Add ones to X
  X <- cbind(rep(1,nrow(X)),X)
  #Response variable
  Y <- as.matrix(Ypred)
  #Intial theta
  initial_theta <- rep(0,ncol(X))
  #Cost at inital theta
  cost(initial_theta)
  
  # Derive theta using gradient descent using optim function
  theta_optim <- optim(par=initial_theta,fn=cost)
  
  #set theta
  theta <- theta_optim$par
  #cost at optimal value of the theta
  theta_optim$value
  
  #prob <- sigmoid(t(c(1,45,85))%*%theta)
  predprobs <- vector(mode="numeric", length=nrow(testdata))
  Y <- cbind(rep(1,nrow(testdata)),testdata)
  for (i in 1:nrow(data))
  {
    y <- Y[i,]
    predprobs[i] <-sigmoid(t(as.vector(y))%*%theta)
    if(predprobs[i] > 0.5)
      predresults[i] <- 1
    else
      predresults[i] <- 0
  }
  return (predresults)
}


setwd("/home/srimugunthan/TookitakiEx") 
#Load data
data <- read.csv("data.csv")
trainX <- as.matrix(data[,c(1,2)])
testX <- as.matrix(data[,c(1,2)])
predresults <- mysimplelogistic(trainX, data$label,testX)

#dses <- data.frame(ses = c("0", "1", "2"), write = mean(ml$write))
#predresults <- mylogisticpredict(bilogisticmodel, newdata = traindf)

results <- table(predresults,data$label)


sink(file = "analysis-output.txt",append=FALSE)
cat("========================MySimple Logistic model========================\n")
cat("---------------------------------------------------------------------\n")
print(results)
sink(NULL)
