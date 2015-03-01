library("matlab")

#http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial
#http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
#http://ufldl.stanford.edu/tutorial/supervised/ExerciseSupervisedNeuralNetwork/
#https://github.com/civilstat/stanford_dl_ex/blob/master/multilayer_supervised/supervised_dnn_cost.m
#https://github.com/amaas/stanford_dl_ex/issues/12

#globals
  # X : the data used to train
  # y : the predicted variable; output column 
  # thisneuralnet parameters (numlayers, layersizes)
  # numfeatures
  # numOutClasses
  # initialtheta
  # the weight matrix (stk)


thisneuralnetw <- list(
  
  numlayers=3,
  numhiddenlayers =2,
  layer_sizes=c(4,4,3),
  num_features=8,
  numoutClasses=3,
  lambda_wtdecay = 1e-5,
  maxiterations = 10
)

stk <- list(
  W <- list(),
  b <- list()
)

#OptTheta 

initialise_weights <- function()
{
      #http://stackoverflow.com/questions/22308937/using-a-single-weight-matrix-for-back-propagation-in-neural-networks?rq=1
      #weights = (2*rand(row_count, col_count)-1) * max_weight;
      #weights <<- 0.005 *matrix(rnorm((row_count * col_count)), 1)
      
      #stk <<- cell(1, numel(thisneuralnetw$layer_sizes));
      for (l in 1 : numel(thisneuralnetw$layer_sizes))
      {
        if (l > 1)
        {
            prev_size = thisneuralnetw$layer_sizes[l-1];
        }
        else
        {
            prev_size = thisneuralnetw$num_features;
        }
        
        cur_size = thisneuralnetw$layer_sizes[l];
        
        
        # Xaxier's scaling factor
        s = sqrt(6) / sqrt(prev_size + cur_size);
        #randmat <- matrix(data = rexp(200, rate = 0.1), nrow = cur_size, ncol = prev_size)
        randmat <- matrix(data = rnorm((cur_size * prev_size), 0.001), nrow = cur_size, ncol = prev_size)
        
        stk$W[[l]] <<- randmat*2*s - s;
        stk$b[[l]] <<- zeros(cur_size, 1);

      }
      return (stk)

}

unroll_theta_to_layers <- function(theta)
{
        #http://stackoverflow.com/questions/1329940/how-do-i-make-a-matrix-from-a-list-of-vectors-in-r
        stk2 <- list(
          W <- list(),
          b <- list()
        )
        
        
        curindex = 1
        for (l in 1 : numel(thisneuralnetw$layer_sizes))
        {
          if (l > 1)
          {
            prev_size = thisneuralnetw$layer_sizes[l-1];
          }
          else
          {
            prev_size = thisneuralnetw$num_features;
          }
          
          cur_size = thisneuralnetw$layer_sizes[l];
          
          
         
          
          Wmat <- matrix(data =theta[curindex:(curindex+(cur_size*prev_size) -1)], nrow = cur_size, ncol = prev_size)
          curindex =curindex+(cur_size*prev_size)
          stk2$W[[l]] <- Wmat;
          bmat <- matrix(data =theta[curindex:(curindex+cur_size-1)], nrow = cur_size, ncol = 1)
          curindex =curindex+(cur_size)
          stk2$b[[l]] <- bmat
          
     }
    return (stk2)
}


roll_layers_to_theta <- function(stkl)
{
        for (l in 1 : numel(thisneuralnetw$layer_sizes))
        {
          if(l == 1)
          {
            theta <-as.vector(stkl$W[[1]])
            theta <-c(theta, as.vector(stkl$b[[1]]))
          }
          else
          {
            theta <-c(theta, as.vector(stkl$W[[l]]))
            theta <-c(theta, as.vector(stkl$b[[l]]))
          }
          
        }
        return (theta)  
}

#function [ cost, grad, pred_prob] = ( theta, ei, data, labels)
mysupervised_nn_cost <- function(theta)
{
      
      
      
      #gradientStk = cell(numHidden+1, 1);
      #hAct = cell(numHidden+1, 1);
      hAct = list()
      stklayer = unroll_theta_to_layers(theta)
      numSamples = nrow(X)
      
      ######################################  
      # # forward propagation
      ######################################
      
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
      }
      
      ######################################
        ## calculate predprobs
      ######################################
      outlayer <- matrix(, ncol = thisneuralnetw$numoutClasses, nrow = 0)
      
      lasthiddenlayer <- hAct[[thisneuralnetw$numhiddenlayers]]
      l <- thisneuralnetw$numlayers
      for(i in 1:nrow(lasthiddenlayer))
      {
        
        activationsperX <- t(as.matrix(rowSums((stk$W[[l]] * (lasthiddenlayer[i,]))) + stk$b[[l]]))
        outlayer <- rbind(outlayer,activationsperX)
      }
      hAct[[l]] = exp(outlayer)
      yh <- hAct[[l]]
      denom <- rowSums(yh)
      pred_prob <- sweep(yh,MARGIN=1,denom,"/")
        
      ######################################
      # cost calculation based on calculated predprobs
      
      ######################################
      #       %% compute cost

      #       y_hat = log(hAct[[numHidden+1]]);
      #       index = sub2ind(size(y_hat), t(labels), 1:numSamples);
      #       cost_term1 = -sum(y_hat(index));
      #       
            
      lgpy <- log(pred_prob)
      cost_term1 = -sum(rowSums(lgpy))
      
      
      ######################################
      # do the back propagation
      ######################################
      #   %% compute gradients using backpropagation
         
      #     % For layers 2:nrLayers, we will need a delta vector
      #   % (so 1st element of deltaStack will just stay empty)
#     
#       deltaStack = cell(nrLayers, 1);
#       #   % The last layer's delta vector is just the residuals,
#       #   % NOT summed over samples like the tutorial mistakenly says.
#       #   % We want one delta per output unit AND per sample.
#       
#       switch ei.output_type
#       case 'categorical'
#           deltaStack{nrLayers} = pred_prob - y_matrix;
#       case 'continuous'
#           deltaStack{nrLayers} = (pred_prob - labels) .* (a{nrLayers}.*(1-a{nrLayers}));
#       end
#       
#   
#       #   % Backpropagate to get deltas for previous layers
#       for l=((nrLayers-1):-1:2)
#       #   % delta{l} = [W{l}' * delta{l+1}] .* [deriv of f wrt z{l}]
#       # % TODO: if generalizing to different function f,
#       # % replace the last product a.*(1-a) with the appropriate derivative of f at z
#           deltaStack{l} = (stack{l}.t(W)*deltaStack{l+1}) .* (a{l}.*(1-a{l}));
#       end
    
      deltastk <- cell(nrLayers, 1)
      deltastk[[nrLayers]] = predprobs -ymatrix
     for(l in (nrLayers-1:2-1))
     {
       fderivative <- hAct[[l]] * (1-hAct[[l]])
       delatstk[[l]] <- ( t(stk$W[[l]]) * deltastk[[l+1]]) * fderivative
     }
      ######################################
      # cost = (cost term 1) + (Weight decay term)
      ######################################
      wCost = 0;
      for (l in  1:(thisneuralnetw$numlayers))
      {
        wCost = wCost + (0.5 * (thisneuralnetw$lambda_wtdecay * sum(rowSums(((stk$W[[l]])^2)))))
      } 
      
      cost = cost_term1 + wCost

      ######################################
      # optional gradient calculation
      ######################################

      #return cost;
      return (cost)
      

  
  
    
}

myNnetTrain <- function( traindata, Ypred)
{
  
  X <<- as.matrix(traindata)
  #X <<- cbind(rep(1,nrow(X)),X)
  Y <<- as.matrix(Ypred)
  
  numClasses <- length(levels(as.factor(Y)))
  num_features <- ncol(X)
  
  thisneuralnetw$num_features <<- num_features
  thisneuralnetw$numoutClasses <<- numClasses
  initialtheta <<- roll_layers_to_theta(initialise_weights())
  #initialtheta <- initialise_weights()
  #initialtheta <<- 0.005 *matrix(rnorm((numClasses * num_features)), 1)
  #initialtheta <<- matrix(initialtheta, nrow = num_features, byrow = TRUE)
  
  
  theta <- roll_layers_to_theta(mysupervised_nn_cost(initialtheta))
   for(iter in 1:(thisneuralnetw$maxiterations))
   {
     theta <- roll_layers_to_theta(mysupervised_nn_cost(theta))
   }
   #theta_optim <- optim(par=initialtheta,fn=mysupervised_nn_cost)
   #OptTheta <<- theta_optim$par
  OptTheta <<- theta 
}

myNnetPredict <- function(testdata)
{
  
      # forward propagation
      
      
      #gradientStk = cell(numHidden+1, 1);
      #hAct = cell(numHidden+1, 1);
      theta <- OptTheta 
      hAct = list()
      stklayer = unroll_theta_to_layers(theta)
      numSamples = nrow(X)
      
      ######################################  
      # forward prop
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
      
        }
        else
        {
          
          m <- matrix(, ncol = thisneuralnetw$layer_sizes[l], nrow = 0)
          for(i in 1:nrow(X))
          {
            activationsperX <- t(as.matrix(rowSums((stk$W[[l]] * (X[i,]))) + stk$b[[l]]))
            m <- rbind(m,activationsperX)
          }
          hAct[[l]] = exp(m)
          
          
        }
      }
      
      
      ## calculate predprobs
      ###########
      outlayer <- matrix(, ncol = thisneuralnetw$numoutClasses, nrow = 0)
      
      lasthiddenlayer <- hAct[[thisneuralnetw$numhiddenlayers]]
      l <- thisneuralnetw$numlayers
      for(i in 1:nrow(lasthiddenlayer))
      {
        
        activationsperX <- t(as.matrix(rowSums((stk$W[[l]] * (lasthiddenlayer[i,]))) + stk$b[[l]]))
        outlayer <- rbind(outlayer,activationsperX)
      }
      hAct[[l]] = exp(outlayer)
      yh <- hAct[[l]]
      denom <- rowSums(yh)
      pred_prob <- sweep(yh,MARGIN=1,denom,"/")
      
      
      
      # return here if pred_only is true
      return (pred_prob)
      
}


dummy_mynnetcall <-  function()
{
 print("dummy nnet call") 
}

start.time <- Sys.time()

end.time <- Sys.time()
Training.time.taken <- end.time - start.time

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
# myNnetTrain(traindata=traindf[1:7],  Ypred= traindf$V8)
# 
# predprobs <- myNnetPredict(testdata = traindf[1:7])
# 
# cum.probs <- t(apply(predprobs,1,cumsum))
# 
# # Draw random values
# vals <- runif(nrow(traindf))
# 
# # Join cumulative probabilities and random draws
# tmp <- cbind(cum.probs,vals)
# 
# # For each row, get choice index.
# k <- ncol(predprobs)
# predresults <- 1 + apply(tmp,1,function(x) length(which(x[1:k] < x[k+1])))
# 
# results <- table(predresults,traindf$V8)
# 
# sink(file = "analysis-output.txt")
# cat("=====================================================\n")
# print(results)
# cat("=====================================================\n")
# sink(NULL)
