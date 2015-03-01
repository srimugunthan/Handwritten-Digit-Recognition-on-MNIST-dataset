function [ stack ] = initialize_weights( ei )

%INITIALIZE_WEIGHTS Random weight structures for a network architecture
%   eI describes a network via the fields layerSizes, inputDim, and outputDim 
%   
%   This uses Xaviers weight initialization tricks for better backprop
%   See: X. Glorot, Y. Bengio. Understanding the difficulty of training 
%        deep feedforward neural networks. AISTATS 2010.

%% initialize hidden layers
stack = cell(1, numel(ei.layer_sizes));
for l = 1 : numel(ei.layer_sizes)
    if l > 1
        prev_size = ei.layer_sizes(l-1);
    else
        prev_size = ei.input_dim;
    end;
    cur_size = ei.layer_sizes(l);
    % Xaxiers scaling factor
s = sqrt(6) / sqrt(prev_size + cur_size);
stack{l}.W = rand(cur_size, prev_size)*2*s - s;
stack{l}.b = zeros(cur_size, 1);
end


####
function stack = params2stack(params, ei)

% Converts a flattened parameter vector into a nice "stack" structure 
% for us to work with. This is useful when youre building multilayer
% networks.
%
% stack = params2stack(params, netconfig)
%
% params - flattened parameter vector
% ei - auxiliary variable containing 
%             the configuration of the network
%


% Map the params (a vector into a stack of weights)
depth = numel(ei.layer_sizes);
stack = cell(depth,1);
% the size of the previous layer
prev_size = ei.input_dim; 
% mark current position in parameter vector
cur_pos = 1;

for d = 1:depth
% Create layer d
stack{d} = struct;

hidden = ei.layer_sizes(d);
% Extract weights
wlen = double(hidden * prev_size);
stack{d}.W = reshape(params(cur_pos:cur_pos+wlen-1), hidden, prev_size);
cur_pos = cur_pos+wlen;

% Extract bias
blen = hidden;
stack{d}.b = reshape(params(cur_pos:cur_pos+blen-1), hidden, 1);
cur_pos = cur_pos+blen;

% Set previous layer size
prev_size = hidden;

end

end
############

function [params] = stack2params(stack)

# % Converts a "stack" structure into a flattened parameter vector and also
# % stores the network configuration. This is useful when working with
# % optimization toolboxes such as minFunc.
# %
# % [params, netconfig] = stack2params(stack)
# %
# % stack - the stack structure, where stack{1}.w = weights of first layer
# %                                    stack{1}.b = weights of first layer
# %                                    stack{2}.w = weights of second layer
# %                                    stack{2}.b = weights of second layer
# %                                    ... etc.
# % This is a non-standard version of the code to support conv nets
# % it allows higher layers to have window sizes >= 1 of the previous layer
# % If using a gpu pass inParams as your gpu datatype
# % Setup the compressed param vector
params = [];


for d = 1:numel(stack)
# % This can be optimized. But since our stacks are relatively short, it
# % is okay
params = [params ; stack{d}.W(:) ; stack{d}.b(:) ];

# % Check that stack is of the correct form
assert(size(stack{d}.W, 1) == size(stack{d}.b, 1), ...
       ['The bias should be a *column* vector of ' ...
        int2str(size(stack{d}.W, 1)) 'x1']);
# % no layer size constrain with conv nets
if d < numel(stack)
assert(mod(size(stack{d+1}.W, 2), size(stack{d}.W, 1)) == 0, ...
       ['The adjacent layers L' int2str(d) ' and L' int2str(d+1) ...
        ' should have matching sizes.']);
end
end

end

###########
function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
# % SUPERVISED_DNN_COST Cost function for supervised neural network
# %   Does all the work of cost / gradient computation
# %   Returns: total cost;
# %     gradient for each W and b at each layer (as a stack);
# %     and prediction probabilities for each class for each sample

# %% default values
po = false;
if exist('pred_only','var')
po = pred_only;
end;

# %% reshape into network
# % stack{1} has weights W and biases b that combine with data X
# %   to form inputs to 1st hidden layer;
# % stack{end} has W and b that combine with last hidden layer
# %   to form output pred_probs (unnormalized)
stack = params2stack(theta, ei);
nrLayers = numel(ei.layer_sizes) + 1;
stackDepth = nrLayers - 1;
a = cell(nrLayers, 1); % 1st layer is data, last is output (unnormalized)
gradStack = cell(stackDepth, 1);
nrSamples = size(data, 2);

# %% forward prop
# %%% YOUR CODE HERE %%%
#   % For layer l = 2:nrLayers
# %   (to be consistent with the tutorial notes:
#        %   with a single hidden layer,
#      %   X is layer 1, hidden is layer 2, output is layer 3;
#      %   so that's a stack depth of 2 = nr of layers BESIDES output)
#      %
     a{1} = data; % a(1)=X;
     for l = 2:nrLayers
     % z(l) = [W*a + b](l-1), for each sample,
     % where b is col.vector with same nr rows as W
     z = bsxfun(@plus, stack{l-1}.W*a{l-1}, stack{l-1}.b);
     if l == nrLayers && strcmp(ei.output_type, 'categorical')
#      % If classifying into categories, then for last layer,
#      % don't do full sigmoid, just exp, to calculate softmax below in pred_prob
     a{l} = exp(z);
     else
#        % For other layers, or for continuous output, DO calculate sigmoid
     a{l} = f(z);
     end
     end
     
#      % IF categorical,
#      % normalize final layer, so each sample's outputs sum to 1,
#      % so that they are actual prediction probabilities.
#      % Final layer outputs are in a, with nrClasses rows and nrSamples cols,
#      % so we need something like (a ./ sum(a,1)), arranged right...
#      % but IF continuous, just use a{nrLayers} as the prediction.
     switch ei.output_type
     case 'categorical'
     pred_prob = bsxfun(@rdivide, a{nrLayers}, sum(a{nrLayers},1));
     case 'continuous'
     pred_prob = a{nrLayers};
     end
     
#      %% return here if only predictions desired.
     if po
     cost = -1;
     grad = [];  
     return;
     end;
     
     
#      %% compute cost
#      %%% YOUR CODE HERE %%%
     
     switch ei.output_type
     case 'categorical'
#      % Find linear equivalents of matrix indices (i,j)
#      % where, in j'th sample (column), i=y(j)
#      % i.e. the row ID is the class of that column
     IDs = sub2ind(size(pred_prob), labels', 1:nrSamples);
#                    % Compute binary matrix where IDs are 1 and rest are 0
                   y_matrix = zeros(size(pred_prob));
                   y_matrix(IDs) = 1;
                   % Cost function f=J(theta), NOT yet including L2 penalty on the weights
                   cost = -sum(log(pred_prob(IDs)));
                   case 'continuous' % use squared error loss, but DON'T divide by NrSamples
                   cost = .5 .* sum(sum((pred_prob - labels).^2));
                   end
                   
                   
                   
                   %% compute gradients using backpropagation
                   %%% YOUR CODE HERE %%%
                     
                     % For layers 2:nrLayers, we will need a delta vector
                   % (so 1st element of deltaStack will just stay empty)
                   deltaStack = cell(nrLayers, 1);
                   % The last layer's delta vector is just the residuals,
                   % NOT summed over samples like the tutorial mistakenly says.
                   % We want one delta per output unit AND per sample.
                   switch ei.output_type
                   case 'categorical'
                   deltaStack{nrLayers} = pred_prob - y_matrix;
                   case 'continuous'
                   deltaStack{nrLayers} = (pred_prob - labels) .* (a{nrLayers}.*(1-a{nrLayers}));
                   end
                   
                   
                   % Backpropagate to get deltas for previous layers
                   for l=((nrLayers-1):-1:2)
                   % delta{l} = [W{l}' * delta{l+1}] .* [deriv of f wrt z{l}]
% TODO: if generalizing to different function f,
% replace the last product a.*(1-a) with the appropriate derivative of f at z
deltaStack{l} = (stack{l}.W'*deltaStack{l+1}) .* (a{l}.*(1-a{l}));
                 end
                 
                 % Compute gradient (sum of contributions of each sample)
                 %   at each layer, for W and b
                 %   (still WITHOUT the L2 penalty on the weights)
                 % NOTE: Take SUMS not MEANS since our loss function is a sum not a mean
                 % (unlike sq.err. loss example in tutorial)
                 % ...and for simplicity, let's also do sums not means for the sq.err. loss too with continuous outputs
                 for l=1:stackDepth
                 % Grad of W{l}(i,j) is matrix of sums of delta{l+1}(i)*a{l}(j) across samples
                 gradStack{l}.W = deltaStack{l+1} * a{l}'; %' % Notepad++ doesn't recognize the 1st ' as transpose
                 % For grad of b, take sum within rows of delta{l+1} i.e. across samples
                 gradStack{l}.b = sum(deltaStack{l+1}, 2);
                 end
                 
                 
                 %% compute L2 weight penalty cost and gradient for non-bias terms
                 %%% YOUR CODE HERE %%%
                   if ei.lambda ~= 0
                 % For each layer with weights W in it...
                 for l=1:stackDepth
                 % Add to the cost: sum of squared weights multiplied by lambda/2
                 cost = cost + (ei.lambda/2)*sum(sum(stack{l}.W .^ 2));
                 % Add to the gradient: weight times lambda
                 gradStack{l}.W = gradStack{l}.W + ei.lambda.*stack{l}.W;
                 end
                 end
                 
                 
                 %% reshape gradients into vector, so gradient search can optimize them all at once
                 [grad] = stack2params(gradStack);
                 end
                 
                 
                 % Helper subfunctions:
                   
                   % Let f be the sigmoid, logistic, or inverse-logit function.
                 % We can apply it element-wise.
                 function h=f(z)
                 h = logsig(z);
                 end
                 
  #######
  
  % runs training procedure for supervised multilayer network
  % softmax output layer with cross entropy loss function
  
  %% setup environment
  % experiment information
  % a struct containing network layer sizes etc
  ei = [];
  
  % add common directory to your path for
  % minfunc and mnist data helpers
  addpath ../common;
  addpath(genpath('../common/minFunc_2012/minFunc'));
  
  %% load mnist data
  [data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();
  
  %% populate ei with the network architecture to train
  % ei is a structure you can use to store hyperparameters of the network
  % the architecture specified below should produce  100% training accuracy
  % You should be able to try different network architectures by changing ei
  % only (no changes to the objective function code)
  
  % dimension of input features
  ei.input_dim = 784;
  % number of output classes
  ei.output_dim = 10;
  % sizes of all hidden layers and the output layer
  ei.layer_sizes = [256, ei.output_dim];
  % scaling parameter for l2 weight regularization penalty
  ei.lambda = 0;
  % which type of activation function to use in hidden layers
  % feel free to implement support for only the logistic sigmoid function
  ei.activation_fun = 'logistic';
  % Whether output is categorical (each sample's y is in 1:k if there are k categories)
                                   % or continuous (each sample's y is a vector in (0,1) range)
  ei.output_type = 'categorical'; % or 'continuous';
  
  %% setup random initial weights
  stack = initialize_weights(ei);
  params = stack2params(stack);
  
  %% setup minfunc options
  options = [];
  options.display = 'iter';
  options.maxFunEvals = 1e6;  % just do a few iters for testing, else set to 1e6
  options.Method = 'lbfgs';
  
  %% run training
  [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
                                                   params,options,ei, data_train, labels_train);
  
  %% compute accuracy on the test and train set
  [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
  [~,pred] = max(pred);
  acc_test = mean(pred'==labels_test);
                  fprintf('test accuracy: %f\n', acc_test);
                  
                  [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
                  [~,pred] = max(pred);
                  acc_train = mean(pred'==labels_train);
  fprintf('train accuracy: %f\n', acc_train);
  
#   % Woohoo! I got it to work with 100% training accuracy, 96.88% test accuracy
#   % (converging in 101 iterations)
#   % with the suggested setup:
#     %   ei.layer_sizes = [256, ei.output_dim];
#   %   ei.lambda = 0;
  
  if false
#   % If instead I set
#   %   ei.lambda = 1;
#   % I get the same results,
#   % so this seems to be an insignificantly small lambda.
#   % To confirm that lambda does have an effect, let's retry with
#   %   ei.lambda = 1000;
#   % but let's restart from the last optimal param values
#   % to hopefully converge a bit faster.
  ei.lambda = 1000;
  [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
                                                   opt_params,options,ei, data_train, labels_train);
  [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
  [~,pred] = max(pred);
  acc_test = mean(pred'==labels_test);
                  fprintf('test accuracy: %f\n', acc_test);
                  [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
                  [~,pred] = max(pred);
                  acc_train = mean(pred'==labels_train);
  fprintf('train accuracy: %f\n', acc_train);
#   % It took longer to converge: stopped after 500 steps
#   % but with train acc 87%, test acc 88%, so not too bad.
#   
#   % Should we try again with intermediate lambda?
  ei.lambda = 10;
  [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
                                                   opt_params,options,ei, data_train, labels_train);
  [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
  [~,pred] = max(pred);
  acc_test = mean(pred'==labels_test);
                  fprintf('test accuracy: %f\n', acc_test);
                  [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
                  [~,pred] = max(pred);
                  acc_train = mean(pred'==labels_train);
  fprintf('train accuracy: %f\n', acc_train);
#   % Again, stopped after max of 500 steps
#   % but with train acc 99%, test acc 98%, so much better.
  
#   % OK, enough of playing with lambda; seems like it's working properly.
#   % Reset to lambda=0;
  ei.lambda = 0;
  
  
#   % Does it also run and work well with fewer units in the hidden layer?
  'Trying a smaller hidden layer of size 100 instead of 256'
  'Using new random-init params'
  ei.layer_sizes = [100, ei.output_dim];
  stack = initialize_weights(ei);
  params = stack2params(stack);
  [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
  params,options,ei, data_train, labels_train);
  [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
  [~,pred] = max(pred);
  acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', acc_test);
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
                 fprintf('train accuracy: %f\n', acc_train);
                 % Yes: converged in 116 steps, train acc 100%, test acc 96%
                 
                 
                 % What about with more hidden layers?
                 'Trying two smaller hidden layers, both of size 100'
                 'Using new random-init params'
                 ei.layer_sizes = [100, 100, ei.output_dim];
                 stack = initialize_weights(ei);
                 params = stack2params(stack);
                 [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
                 params,options,ei, data_train, labels_train);
                 [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
                 [~,pred] = max(pred);
                 acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', acc_test);
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
                 fprintf('train accuracy: %f\n', acc_train);
                 % Yes! It works! Awesome :)
                 % Converged in 146 steps, train acc 100%, test acc 96%
                 % So the extra layer didn't really help it,
                 % BUT at least the code clearly works with the extra layer.
                 
                 % Awesome. Now can we try autoencoder?
                 % (Had to modify the supervised_dnn_cost.m code
                    %  because our softmax regression assumes that y is a vector, not a matrix:
                      %  for example e, y(e) = k = which class example e belongs to,
                    %  but X(:,e) is a continuous vector with values in (0,1) range.
                    %  But that's done now!)
                    
                    
                    % The simplest interesting DL things to try would be:
                    % Run optimization once with X as both the inputs AND outputs,
                    %   then use those estimates as the initial param values
                    %   and rerun optimization with Y as the output.
                    % How does it affect train and test accuracy?
                    % How does it change with nr and sizes of hidden layers?
                    'Trying autoencoder: Can we get X back with one hidden layer of size 100?'
                    ei.output_type = 'continuous';
                    ei.output_dim = ei.input_dim;
                    ei.layer_sizes = [100, ei.output_dim];
                    stack = initialize_weights(ei);
                    params = stack2params(stack);
                    [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
                    params,options,ei, data_train, data_train);
                    % It didn't finish converging -- just ran to max of 500 iterations
                    [~, ~, pred_train] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
                    [~, ~, pred_test] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
                    train_rms=sqrt(mean(mean((pred_train - data_train).^2)));
                    fprintf('RMS training error: %f\n', train_rms);
                    test_rms=sqrt(mean(mean((pred_test - data_test).^2)));
                    fprintf('RMS testing error: %f\n', test_rms);
                    % RMS train and test error are both around 0.062
                    % while mean of train and test data are around 0.13
                    % so the prediction RMS isn't TOTALLY hopeless.
                    'Saving the opt_params from this autoencoder'
                    opt_params_autoencoder = opt_params;
                    opt_stack_autoencoder = params2stack(opt_params_autoencoder, ei);
                    
                    % Try to visualize the inputs and predictions:
                    figure(1); imagesc(reshape(data_train(:,1), 28, 28))
                    figure(2); imagesc(reshape(pred_train(:,1), 28, 28))
                    figure(3); imagesc(reshape(data_test(:,1), 28, 28))
                    figure(4); imagesc(reshape(pred_test(:,1), 28, 28))
                    % Yep, the reconstructions look pretty decent,
                    % even though it didn't finish converging.
                    
                    'Now trying to use those parameter estimates from P(X|X)'
                    'as intial values for supervised learning of P(Y|X)'
                    ei.output_type = 'categorical';
                    ei.output_dim = 10;
                    ei.layer_sizes = [100, ei.output_dim];
                    stack = initialize_weights(ei);
                    % TODO: how to modify the weights in this stack
                    %       so that the bottom-most weights are the ones
                    %       trained by the autoencoder of X?
                    stack{1} = opt_stack_autoencoder{1};
                    params = stack2params(stack);
                    [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
                                                                     params,options,ei, data_train, labels_train);
                    [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
                    [~,pred] = max(pred);
                    acc_test = mean(pred'==labels_test);
                                    fprintf('test accuracy: %f\n', acc_test);
                                    [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
                                    [~,pred] = max(pred);
                                    acc_train = mean(pred'==labels_train);
                    fprintf('train accuracy: %f\n', acc_train);
                    % Converged in 141 steps, train acc 100%, test acc 95.51%
                    % so the pre-training did NOT help the test accuracy at all,
                    % but it DID help the train acc over the 100-unit version
                    % (though it's no better than the default 256-unit version at the top)
                       
                       
                       % So now we have opt_params from the pre-trained AND supervised network,
                       % as well as opt_params_autoencoder from the pre-training run alone.
                       % Can we visualize the inputs that maximize each of the hidden layer units?
                       %   The tutorial says to rescale the weights, but it seems like by a sum which is constant over pixels...?
                       %   Yes, but it's a different scaling factor for each UNIT,
                       %   so we might as well rescale them for simultaneous display's sake.
                       % Let's show 2 images from the autoencoder,
                       % and 2 from the fully trained network:
                         opt_stack = params2stack(opt_params, ei);
                       autoencW = zeros(100,784);
                       fulltrainW = zeros(100,784);
                       % Row: which unit?
                       % Col: which pixel within that unit?
                       for i=1:100
                       autoencW(i,:) = opt_stack_autoencoder{1}.W(i,:);
                       autoencW(i,:) = autoencW(i,:) / sum(autoencW(i,:).^2);
                       fulltrainW(i,:) = opt_stack{1}.W(i,:);
                       fulltrainW(i,:) = fulltrainW(i,:) / sum(fulltrainW(i,:).^2);
                       end
                       autoencMin = min(min(autoencW));
                       autoencMax = max(max(autoencW));
                       fulltrainMin = min(min(fulltrainW));
                       fulltrainMax = max(max(fulltrainW));
                       for i=1:100
                       figure(1)
                       subplot(10,10,i)
                       imagesc(reshape(autoencW(i,:), 28, 28))
                       set(gca,'xtick',[]); set(gca,'ytick',[]); caxis([autoencMin autoencMax])
                       figure(2)
                       subplot(10,10,i)
                       imagesc(reshape(fulltrainW(i,:), 28, 28))
                       set(gca,'xtick',[]); set(gca,'ytick',[]); caxis([fulltrainMin fulltrainMax])
                       end
                       % Fascinating! When we look at the autoencoder in Fig 1,
                       % a few look like real signal and the rest look like white noise.
                       % Then, looking at the fully-trained Fig 2,
                       % all those that looked good in Fig 1 still look the same,
                       % while those that looked like noise in Fig 1 look much less noisy
                       % (though still not as clear as the original "good" ones).
                       % ...
                       % although that's when we let each subplot scale to its own min and max.
                       % When we scale all subplots within a figure to the same min and max,
                       % the units in Fig 2 that looks like white noise in Fig 1
                       % now just look like almost nothing at all --
                       % they've been smoothed spatially, but also dampened a lot towards 0 overall.
                       % ...
                       % What if instead of min and max, we use +/- of max(abs(min,max))?
                       % Then at least we know that green is neutral.
                       %   autoencMM = max(abs(autoencMin), abs(autoencMax));
                       %   fulltrainMM = max(abs(fulltrainMin), abs(fulltrainMax));
                       %   caxis([-autoencMM autoencMM]);
                       %   caxis([-fulltrainMM fulltrainMM]);
                       % Nope, that doesn't look great either. Oh well.
                       % ...
                       % I need to find a better way to plot these:
                       % have a neutral color at 0, and diverging colors towards the + and - extremes,
                       % such that the "good" plots look similar in BOTH figures.
                       % But that's tough, since the max in figure 1 is in a white-noise subplot, not a "good" subplot,
                       % so we can't just use the max of the data to scale things. Argh.
                       
                       
                       % ANYWAY! The autoencoder seems to work, and seems to do something productive :)
                       % 
                       % Last step at this stage should be to rerun the training WITHOUT pretraining,
                       % and visualize THOSE hidden units too, and see if any of them look
                       % as good as 'the good ones' from the pretrained network
                       % (on the same set of train/test data, just with new random-init params).
                       % 
                       opt_params_fulltrain = opt_params;
                       opt_stack_fulltrain = opt_stack;
                       'Now retrying supervised learning of P(Y|X), with NO pretraining'
                       ei.output_type = 'categorical';
                       ei.output_dim = 10;
                       ei.layer_sizes = [100, ei.output_dim];
                       stack = initialize_weights(ei);
                       params = stack2params(stack);
                       [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
                       params,options,ei, data_train, labels_train);
                       [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
                       [~,pred] = max(pred);
                       acc_test = mean(pred'==labels_test);
                    fprintf('test accuracy: %f\n', acc_test);
                    [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
                    [~,pred] = max(pred);
                    acc_train = mean(pred'==labels_train);
                                     fprintf('train accuracy: %f\n', acc_train);
                                     % Converged in 122 steps, train acc 100%, test acc 96.3%
                                     opt_params_justtrain = opt_params;
                                     opt_stack_justtrain = params2stack(opt_params_justtrain, ei);
                                     justtrainW = zeros(100,784);
                                     % Row: which unit?
                                     % Col: which pixel within that unit?
                                     for i=1:100
                                     justtrainW(i,:) = opt_stack_justtrain{1}.W(i,:);
                                     justtrainW(i,:) = justtrainW(i,:) / sum(justtrainW(i,:).^2);
                                     end
                                     for i=1:100
                                     figure(3)
                                     subplot(10,10,i)
                                     imagesc(reshape(justtrainW(i,:), 28, 28))
                                     set(gca,'xtick',[]); set(gca,'ytick',[]);
                                     end
  % OK, it's really hard to tell (without good colormap)
	% whether this is just noise or actually something useful. Argh.
	% But I *THINK* it looks different at least in the sense that
	% there are none of those "good"-looking units that we got in pretraining.
	% Maybe the pretrainer just needs to run longer?
	
	% FINALLY, we could also redo the sq.err. loss and the gradient
	% into KL-divergence loss, which requires changing the cost penalty
	% and also the delta terms in the gradient...


	% Then we can move to the next stage, of trying several different network architectures,
	% with and without pretraining, so we can test the hypotheses from the paper.
	% Also try SUPERVISED pretraining and see if it differs much.
end

#############


