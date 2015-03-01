function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)
{
    # numClasses - the number of classes 
    # inputSize - the size N of the input vector
    # lambda - weight decay parameter
    # data - the N x M input matrix, where each column data(:, i) corresponds to
    #        a single test set
    # labels - an M x 1 matrix containing the labels corresponding for the input data
    
    
    # Unroll the parameters from theta
    theta = reshape(theta, numClasses, inputSize);
    
    numCases = size(data, 2);
    
    groundTruth = full(sparse(labels, 1:numCases, 1));
    cost = 0;
    
    thetagrad = zeros(numClasses, inputSize);
    
    #  Instructions: Compute the cost and gradient for softmax regression.
    #                You need to compute thetagrad and cost.
    #                The groundTruth matrix might come in handy.
    
    
    # size(groundTruth) % 10 100
    # size(theta)       % 10 8
    # size(labels)      % 100 1
    # size(data)        % 8 10
    # size (thetagrad)  % 10 8  
    # h                 % 10 10
    
    y = groundTruth;
    m = numCases;
    
    # note that if we subtract off after taking the exponent, as in the
    # text, we get NaN
    
    td = theta * data;
    td = bsxfun(@minus, td, max(td));
    temp = exp(td);
    
    denominator = sum(temp);
    p = bsxfun(@rdivide, temp, denominator);
    cost = (-1/m) * sum(sum(y .* log(p))) + (lambda / 2) * sum(sum(theta .^2));
    
    #thetagrad = (-1/m) * (y - p) * data' + lambda * theta;
    
    
    # Unroll the gradient matrices into a vector for minFunc
    grad = [thetagrad(:)];

}



function [softmaxModel] = softmaxTrain(inputSize, numClasses, lambda, inputData, labels, options)
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
    
    if ~exist('options', 'var')
      options = struct;
    end
    
    if ~isfield(options, 'maxIter')
      options.MaxIter = 400;
    end
    
    # initialize parameters
    theta = 0.005 * randn(numClasses * inputSize, 1);
    
    # Use minFunc to minimize the function
    addpath ../common/fminlbfgs
    options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost

    # function. Generally, for minFunc to work, you
    # need a function pointer with two outputs: the
    # function value and the gradient. In our problem,
    # softmaxCost.m satisfies this

    options.Display = 'iter';
    options.GradObj = 'on';
    
    [softmaxOptTheta, cost] = fminlbfgs( @(p) softmaxCost(p, ...
    numClasses, inputSize, lambda, ...
    inputData, labels), ...                                   
    theta, options);
    
    # Fold softmaxOptTheta into a nicer format

    softmaxModel.optTheta = reshape(softmaxOptTheta, numClasses, inputSize);
    softmaxModel.inputSize = inputSize;
    softmaxModel.numClasses = numClasses;
    
    
}

function [pred] = softmaxPredict(softmaxModel, data)
{

    # softmaxModel - model trained using softmaxTrain
    # data - the N x M input matrix, where each column data(:, i) corresponds to
    #        a single test set
    #
    # Your code should produce the prediction matrix 
    # pred, where pred(i) is argmax_c P(y(c) | x(i)).
     
    # Unroll the parameters from theta
    theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
    pred = zeros(1, size(data, 2));
    
    
    #  Instructions: Compute pred using theta assuming that the labels start 
    #                from 1.
    
    
    
    size(pred); %   1 10000
    size(data); % 784 10000
    size(theta);%  10     8
    
    p = theta*data;
    [junk, idx] = max(p, [], 1);
    
    pred = idx;
    
}


if DEBUG
numGrad = computeNumericalGradient( @(x) softmaxCost(x, numClasses, ...
                                                     inputSize, lambda, inputData, labels), theta);

# Use this to visually compare the gradients side by side
disp([numGrad grad]); 

# Compare numerically computed gradients with those computed analytically
diff = norm(numGrad-grad)/norm(numGrad+grad);
disp(diff); 
# The difference should be small. 
# In our implementation, these values are usually less than 1e-7.

# When your gradients are correct, congratulations!
end




#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  softmax exercise. You will need to write the softmax cost function 
#  in softmaxCost.m and the softmax prediction function in softmaxPred.m. 
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#  (However, you may be required to do so in later exercises)

# ======================================================================
# STEP 0: Initialise constants and parameters
#
#  Here we define and initialise some constants which allow your code
#  to be used more generally on any arbitrary input. 
#  We also initialise some parameters used for tuning the model.

inputSize = 28 * 28; % Size of input vector (MNIST images are 28x28)
numClasses = 10;     % Number of classes (MNIST images fall into 10 classes)

lambda = 1e-4; % Weight decay parameter


# STEP 1: Load data
#
#  In this section, we load the input and output data.
#  For softmax regression on MNIST pixels, 
#  the input data is the images, and 
#  the output data is the labels.
#

# Change the filenames if youve saved the files under different names
# On some platforms, the files might be saved as 
# train-images.idx3-ubyte / train-labels.idx1-ubyte

images = loadMNISTImages('mnist/train-images-idx3-ubyte');
labels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

inputData = images;

# For debugging purposes, you may wish to reduce the size of the input data
# in order to speed up gradient checking. 
# Here, we create synthetic dataset using random data for testing

DEBUG = false; % Set DEBUG to true when debugging.
if DEBUG
inputSize = 8;
inputData = randn(8, 100);
labels = randi(10, 100, 1);
end

% Randomly initialise theta
theta = 0.005 * randn(numClasses * inputSize, 1);

##======================================================================
# STEP 2: Implement softmaxCost
#
#  Implement softmaxCost in softmaxCost.m. 

[cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels);

#======================================================================
# STEP 3: Gradient checking
#
#  As with any learning algorithm, you should always check that your
#  gradients are correct before learning the parameters.
# 


#======================================================================
# STEP 4: Learning parameters
#
#  Once you have verified that your gradients are correct, 
#  you can start training your softmax regression code using softmaxTrain
#  (which uses minFunc).

options.MaxIter = 100;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData, labels, options);

# Although we only use 100 iterations here to train a classifier for the 
# MNIST data set, in practice, training for more iterations is usually
# beneficial.

#======================================================================
# STEP 5: Testing
#
#  You should now test your model against the test images.
#  To do this, you will first need to write softmaxPredict
#  (in softmaxPredict.m), which should return predictions
#  given a softmax model and the input data.

images = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
labels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

inputData = images;

# You will have to implement softmaxPredict in softmaxPredict.m
[pred] = softmaxPredict(softmaxModel, inputData);

acc = mean(labels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);

# Accuracy is the proportion of correctly classified images
# After 100 iterations, the results for our implementation were:
#
# Accuracy: 92.200%
#
# If your values are too low (accuracy less than 0.91), you should check 
# your code for errors, and make sure you are training on the 
# entire data set of 60000 28x28 training images 
# (unless you modified the loading code, this should be the case)