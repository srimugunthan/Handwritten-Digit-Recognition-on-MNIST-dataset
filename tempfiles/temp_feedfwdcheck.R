library("matlab")
library("matrixcalc")
# i <- c(1,3:8)
# j <- c(2,9,6:10)
# x <- 7 * (1:7)
# X <- sparseMatrix(i, j, x = x)
# ################################################
# 
# ###########################################
# 
# #y_hat = exp(theta' * X); 
# yhat = exp(t(theta)*X)
# 
# 
# #y_hat = [y_hat; ones(1, size(y_hat, 2))]; 
# yhat = rbind(yhat,array(1,ncol(yhat)))
# 
# #y_hat_sum = sum(y_hat, 2); % K * 1
# yhatsum = sum(yhat,2)
# 
# 
# y_hat_sum(end, :) = 1; % K * 1
# #p_y = bsxfun(@rdivide, y_hat, y_hat_sum); % K * m
# p_y = bsxfun("/", yhat, yhatsum);
# 
# A = log(p_y); % K * m
# A = log(p_y)
# 
# index = sub2ind(size(y_hat), y, 1 : size(y_hat, 2));
# A(end, :) = 0;
# A(index); % m * 1
# f = -sum(A(index));
# indicator = zeros(size(p_y)); % K * m
# indicator(index) = 1;
# g = -X * (indicator - p_y)'; % K * n
#             
# g=g(:, 1:end - 1); 
# g = g(:); % make gradient a vector for minFunc
# 
# 


# mat = matrix( c(3, 4, 6, 8, 9,  16), nrow=2, ncol=3)
# vec = c(3,4)
# #ans <- apply(mat, 2, "/", vec)
# ans <- sweep(mat,MARGIN=1,vec,"/")
# print(mat[1,2])

# NUM <- 1000 # NUM is how many objects I want to have
# xVal <- vector(NUM, mode="list")
# yVal <- vector(NUM, mode="list")
# title   <- vector(NUM, mode="list")
# for (i in 1:NUM) {
#   xVal[i]<-list(rnorm(50))
#   yVal[i]<-list(rnorm(50))
#   title[i]<-list(paste0("This is title for instance #", i))
# }
# myObject <- list(xValues=xVal, yValues=yVal, titles=title)
# # now I can address any member, as needed:
# print(myObject$titles[[3]])
# print(myObject$xValues[[4]]) 
# 
# roll_into_theta <- function(stk)
# {
#   for (l in 1 : 3)
#   {
#     if(l == 1)
#     {
#       theta <-as.vector(stk$W[[1]])
#       theta <-c(theta, as.vector(stk$b[[1]]))
#     }
#     else
#     {
#       theta <-c(theta, as.vector(stk$W[[l]]))
#       theta <-c(theta, as.vector(stk$b[[l]]))
#     }
#     
#   }
#   return (theta)  
# }
# 


# unroll_into_layers <- function(theta)
# {
#   #http://stackoverflow.com/questions/1329940/how-do-i-make-a-matrix-from-a-list-of-vectors-in-r
#   
#   stk2 <- list(
#     W <- list(),
#     b <- list()
#   )
#   
#   curindex = 1
#   for (l in 1 : 3)
#   {
#     if (l > 1)
#     {
#       prev_size = 2
#     }
#     else
#     {
#       prev_size = 2
#     }
#     
#     cur_size = 4
#     
#     Wmat <- matrix(data =theta[curindex:(curindex+(cur_size*prev_size) -1)], nrow = 4, ncol = 2)
#     curindex =curindex+(cur_size*prev_size)
#     stk2$W[[l]] <<- Wmat;
#     bmat <- matrix(data =theta[curindex:(curindex+cur_size-1)], nrow = cur_size, ncol = 1)
#     curindex =curindex+(cur_size)
#     stk2$b[[l]] <<- bmat
#     
#   }
#   return(stk2)
# }






#C[[2]] <- array(0,dim=c(3,3))

#df <- data.frame(W=vector("list"))
#s = sqrt(6) / sqrt(4 + 2);
#randmat <- matrix(data = rexp(200, rate = 0.1), nrow = cur_size, ncol = prev_size)

#df[1]$W <- as.vector(as.list(W));
#stack{l}.b
 
# df[1][2] <- as.list(b);
#for(i in 1:3)
# {
#   randmat <- matrix(data = rnorm((7 * 4), 0.001), nrow = 7, ncol = 4)
#   Wmat <- (randmat*2*s - s);
#   b <- zeros(7, 1)
#   stk$W[[1]] <- Wmat
#   stk$b[[1]] <- b
#   
#   randmat <- matrix(data = rnorm((4 * 3), 0.001), nrow = 4, ncol = 3)
#   Wmat <- (randmat*2*s - s);
#   b <- zeros(4, 1)
#   stk$W[[2]] <- Wmat
#   stk$b[[2]] <- b
# }

#theta <<- roll_into_theta(stk)
#tstk2 <<- unroll_into_layers(theta)
#C[[2]] <- W
#append(C,W)


# hA <- list()
# for (l in 1 : 2)
# {
# #   if (l > 1)
# #   {
# #     hA[[l]] = (( stk$W[[l]]*  hA[[l - 1]])) 
# #     #%+% kronecker(matrix(1,1,numSamples),stk$b[[l]]);
# #   }
# #   else
#   {
#     print(dim(stk$W[[l]]))
#     print(dim(X))
#     
#     hA[[l]] = ((stk$W[[l]]* X )) 
#     #%+% kronecker(matrix(1,1,numSamples),stk$b[[l]]);
#   }
# }
# 

thisneuralnetw <- list(
  
  numlayers=3,
  numhiddenlayers =2,
  layer_sizes=c(4,4,3),
  num_features=8,
  numoutClasses=3
  
)



stk <<- list(
  W <- list(),
  b <- list()
)




setwd("/home/srimugunthan/TookitakiEx")
seeds <-  read.csv("seeds_dataset.txt",   header=FALSE, sep="", as.is=TRUE)
seedstrain<- sample(1:210,147)
seedstest <- setdiff(1:210,seedstrain)
traindf <- seeds[seedstrain,]
testdf <-  seeds[seedstest,]
# 
# X <<- traindf[1:7]
# Y <<- traindf$V8
# 
# numSamples <- ncol(X)
# 

#v1 <- c(2,2,2,2,2,2,2,2)
#v2 <- c(2,2,2,2,2,2,2,2)
m <- as.matrix(rbind(c(1,1,1,1,1,1,1,1),c(2,2,2,2,2,2,2,2)))
m <- m[rep(1:nrow(m), times = 50), ]
X <<- m
numSamples <- ncol(X)

for (l in 1 : numel(thisneuralnetw$layer_sizes))
{
  if (l > 1)
  {
    prev_size = thisneuralnetw$layer_sizes[l-1];
    cur_size = thisneuralnetw$layer_sizes[l];
    
    
    # Xaxier's scaling factor
    
    #randmat <- matrix(data = rexp(200, rate = 0.1), nrow = cur_size, ncol = prev_size)
    randmat <- matrix(data = rep.int(2, cur_size*prev_size) , nrow = cur_size, ncol = prev_size)
    
    stk$W[[l]] <- randmat;
    stk$b[[l]] <-  matrix(data = rep.int(2, cur_size) , nrow = cur_size, ncol = 1)
    
  }
  else
  {
    prev_size = thisneuralnetw$num_features;
    cur_size = thisneuralnetw$layer_sizes[l];
    
    
    # Xaxier's scaling factor
    
    #randmat <- matrix(data = rexp(200, rate = 0.1), nrow = cur_size, ncol = prev_size)
    randmat <- matrix(data = rep.int(-2, cur_size*prev_size) , nrow = cur_size, ncol = prev_size)
    
    stk$W[[l]] <- randmat;
    #stk$b[[l]] <- zeros(cur_size, 1);
    stk$b[[l]] <-  matrix(data = rep.int(2, cur_size) , nrow = cur_size, ncol = 1)
    
  }
  
 
}


hAct <- list()
for (l in 1 : thisneuralnetw$numhiddenlayers)
{
  if (l > 1)
  {
    m <- matrix(, ncol = thisneuralnetw$layer_sizes[l], nrow = 0)
    prevlayer <-   hAct[[l - 1]]
    for(i in 1:nrow(prevlayer))
    {
    
    activationsperX <- t(as.matrix(rowSums((stk$W[[l]] * (prevlayer[i,]))) + stk$b[[l]]))
    m <- rbind(m,activationsperX)
    }
    hAct[[l]] = exp(m)
    #hAct[[l]] = (stk$W[[l]] * hAct[[l - 1]])
    #+ kronecker(matrix(1,1,numSamples),stklayer$b[[l]]);
  }
  else
  {
    
    print(dim(stk$W[[l]]))
    print(dim(X))
    m <- matrix(, ncol = thisneuralnetw$layer_sizes[l], nrow = 0)
    for(i in 1:nrow(X))
    {
      activationsperX <- t(as.matrix(rowSums((stk$W[[l]] * (X[i,]))) + stk$b[[l]]))
      m <- rbind(m,activationsperX)
    }
    hAct[[l]] = exp(m)
    
    #hAct[[l]] = (stk$W[[l]] * t(X[1,]))
    #+  kronecker(matrix(1,1,numSamples),stklayer$b[[l]]);
  }
  #hAct{l} = softmax(hAct{l}, 1); 
  #hAct[[l]] <- exp(hAct[[l]])
  #hAct{l} = sigmoid(hAct{l});
  
  #hAct{l} = tanh(hAct{l});
  
  
}

outlayer <- matrix(, ncol = thisneuralnetw$numoutClasses, nrow = 0)

lasthiddenlayer <- hAct[[thisneuralnetw$numhiddenlayers]]
l <- thisneuralnetw$numlayers
for(i in 1:nrow(lasthiddenlayer))
{
  
  activationsperX <- t(as.matrix(rowSums((stk$W[[l]] * (lasthiddenlayer[i,]))) + stk$b[[l]]))
  outlayer <- rbind(outlayer,activationsperX)
}
hAct[[l]] = outlayer

