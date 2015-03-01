

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


bilogisticmodel <- mysimplelogistic(f, data = traindf)

#dses <- data.frame(ses = c("0", "1", "2"), write = mean(ml$write))
predresults <- mylogisticpredict(bilogisticmodel, newdata = traindf)

mysimplelogistic <- function(formula, data)
{
  
}


setwd("/home/srimugunthan/TookitakiEx") 
#Load data
data <- read.csv("data.csv")
#Create plot
#plot(data$score.1,data$score.2,col=as.factor(data$label),xlab="Score-1",ylab="Score-2")
#Predictor variables
X <- as.matrix(data[,c(1,2)])
#Add ones to X
X <- cbind(rep(1,nrow(X)),X)
#Response variable
Y <- as.matrix(data$label)
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
predprobs <- vector(mode="numeric", length=nrow(data))
for (i in 1:nrow(data))
{
  y <- c(1, data[i,1],data[i,2])
  predprobs[i] <-sigmoid(t(as.vector(y))%*%theta)
  if(predprobs[i] > 0.5)
    predresults[i] <- 1
  else
    predresults[i] <- 0
}


# cum.probs <- t(apply(predprobs,1,cumsum))
# # Draw random values
# vals <- runif(nrow(data))
# # Join cumulative probabilities and random draws
# tmp <- cbind(cum.probs,vals)
# # For each row, get choice index.
# k <- ncol(predprobs)
# predresults <- 1 + apply(tmp,1,function(x) length(which(x[1:k] < x[k+1])))
results <- table(predresults,data$label)
View(results)

