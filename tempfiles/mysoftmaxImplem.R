library("pracma")
library('Matrix')

#returns cost and gradient
#http://blog.csdn.net/kylinxu70/article/details/17284343
softmaxCost <- function(theta, numClasses, inputSize, lambda, data, labels)
{
    
      # Unroll the parameters from theta
      #theta = reshape(theta, numClasses, inputSize);
      theta = matrix(theta, numClasses, inputSize);
      
      #numCases = size(data, 2);
      numCases = ncol(data);
      
      #groundTruth = full(sparse(labels, 1:numCases, 1));
      #https://stat.ethz.ch/R-manual/R-devel/library/Matrix/html/sparseMatrix.html
      #http://in.mathworks.com/help/matlab/ref/sparse.html?searchHighlight=sparse
      #https://stat.ethz.ch/R-manual/R-devel/library/Matrix/html/sparseMatrix.html
      #http://stackoverflow.com/questions/17375056/r-sparse-matrix-conversion
      #http://www.johnmyleswhite.com/notebook/2011/10/31/using-sparse-matrices-in-r/
      groundTruth <- as.matrix(sparseMatrix(i= labels, j=seq(1:numCases), x=1))
      
      
      cost = 0;
      
      #thetagrad = zeros(numClasses, inputSize);
      thetagrad = array(0,c(numClasses,inputSize))
      
      #  Instructions: Compute the cost and gradient for softmax regression.
      #                You need to compute thetagrad and cost.
      #                The groundTruth matrix might come in handy.
      
      
      
      y = groundTruth;
      m = numCases;
      
      # note that if we subtract off after taking the exponent, as in the
      # text, we get NaN
      
      td = theta * data;
      td = bsxfun("-", td, max(td));
      temp = exp(td);
      
      denominator = sum(temp);
      p = bsxfun("/", temp, denominator);
      
      
      
      cost = (-1/m) * sum(sum(y*log(p))) + (lambda / 2) * sum(sum(theta ^2));
      thetagrad = (-1/m) * (y - p) * t(data) + lambda * theta;
      
      # Unroll the gradient matrices into a vector for minFunc
      #grad = [thetagrad(:)];
      grad = rbind(as.vector(thetagrad))
      
      return (list(cost=cost, grad=grad))
      
}

softmaxTrain <- function(inputSize, numClasses, lambda, inputData, labels, options)
{
  #softmaxTrain Train a softmax model with the given parameters on the given
  # data. Returns softmaxOptTheta, a vector containing the trained parameters
  # for the model.
  
  
  
  # inputSize: the size of an input vector x^(i)
  # numClasses: the number of classes 
  # lambda: weight decay parameter
  # inputData: an N by M matrix containing the input data, such that
  #            inputData(:, c) is the cth input
  # labels: M by 1 matrix containing the class labels for the
  #            corresponding inputs. labels(c) is the class label for
  #            the cth input
  # options (optional): options
  #   options.maxIter: number of iterations to train for
  
  #options = struct;
  #options.MaxIter = 400;
  
  # initialize parameters
  #theta = 0.005 * randn(numClasses * inputSize, 1);
  theta = 0.005 *matrix(rnorm((numClasses * inputSize)), 1)
  
  # Use minFunc to minimize the function
  #addpath ../common/fminlbfgs
  #options.Method = 'lbfgs';
  # Here, we use L-BFGS to optimize our cost
  # function. Generally, for minFunc to work, you
  # need a function pointer with two outputs: the
  # function value and the gradient. In our problem,
  # softmaxCost.m satisfies this
  
  
#   options.Display = 'iter';
#   options.GradObj = 'on';
#   
#   #[softmaxOptTheta, cost] = fminlbfgs( @(p) softmaxCost(p, ...
#                                                         numClasses, inputSize, lambda, ...
#                                                         inputData, labels), ...                                   
#                                        theta, options);
# 


#http://stackoverflow.com/questions/23718161/optimization-of-a-function-in-r-l-bfgs-b-needs-finite-values-of-fn
#https://stat.ethz.ch/R-manual/R-devel/library/stats/html/optim.html
#http://cran.r-project.org/web/packages/lbfgs/vignettes/Vignette.pdf
#http://grokbase.com/t/r/r-help/1097ev4z34/r-question-on-optim
#http://de.lyra.nom.br:8180/doc/r-recommended/library/MASS/scripts/ch16.R

  softmaxOptTheta <- optim(par=theta,fn=softmaxCost,method = "L-BFGS-B")
  # Fold softmaxOptTheta into a nicer format
  
  #softmaxModel.optTheta = reshape(softmaxOptTheta, numClasses, inputSize);
  softmaxModel.optTheta <- matrix(softmaxOptTheta, numClasses, inputSize);


  softmaxModel.inputSize <- inputSize;
  softmaxModel.numClasses <- numClasses;
  
  return (softmaxModel)
}


softmaxPredict <- function(softmaxModel, testdata)
{
  
  # softmaxModel - model trained using softmaxTrain
  # data - the N x M input matrix, where each column data(:, i) corresponds to
  #        a single test set
  #
  # Your code should produce the prediction matrix 
  # pred, where pred(i) is argmax_c P(y(c) | x(i)).
  
  # Unroll the parameters from theta
  #theta = softmaxModel.optTheta;
  theta <- softmaxModel.optTheta$par
  
  # this provides a numClasses x inputSize matrix
  #pred = zeros(1, size(data, 2));
  predprobs <- vector(mode="numeric", length=nrow(testdata))
  
  Y <- cbind(rep(1,nrow(testdata)),testdata)
  
  #size(pred); #   1 10000
  #size(data); # 784 10000
  #size(theta); #  10     8
  
  for (i in 1:nrow(testdata))
  {
    #  Instructions: Compute pred using theta assuming that the labels start 
    #                from 1.
    
    
    
    #p = theta*data;
    y <- Y[i,]
    predprobs[i] <-sigmoid(t(as.vector(y))%*%theta)
    
    #[junk, idx] = max(p, [], 1);
    
    #pred = idx;
  
  }
  return (predprobs)
  
  
  
}


# Compiled versions
require(compiler)
compiled_code <- cmpfun(softmaxPredict)
