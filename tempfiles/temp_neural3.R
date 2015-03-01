
function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
{
      # %SPNETCOSTSLAVE Slave cost function for simple phone net
      # %   Does all the work of cost / gradient computation
      # %   Returns cost broken into cross-entropy, weight norm, and prox reg
      # %        components (ceCost, wCost, pCost)
      # 
      # %% default values
      po = false;
      if exist('pred_only','var')
        po = pred_only;
      end;
      
      #       %% reshape into network
      numHidden = numel(ei.layer_sizes) - 1;
      numSamples = ncol(data);
      
      hAct = cell(numHidden+1, 1);
      gradStack = cell(numHidden+1, 1);

      # One can think of a neuralma network as a “stack” of models. 
      # On the bottom of the stack are the original features.
      # From these features are learned a variety of relatively simple models.
      # Let’s say these are logistic regressions. Then, each subsequent layer 
      # the stack applies a simple model (let’s say, another logistic regression)
      # to the outputs of the next layer down. So in a two-layer stack,
      # we would learn a set of logistic regressions from the original features,
      # and then learn a logistic regression using as features the outputs of the
      # first set of logistic regressions

      stack = params2stack(theta, ei);

      
      
      
      ######################################  
      #       %% forward prop
      for (l in 1 : numHidden)
      {
          if l > 1
              hAct{l} = stack{l}.W * hAct{l - 1} + kronecker(matrix(1,1,numSamples),stack{l}.b);
          else
    
              hAct{l} = stack{l}.W * data + kronecker(matrix(1,1,numSamples),stack{l}.b);
          end
          #hAct{l} = sigmoid(hAct{l});
          #hAct{l} = softmax(hAct{l}, 1);
          hAct{l} = tanh(hAct{l});
          
    
      }
      
      l = numHidden+1;
      y_hat = stack{l}.W * hAct{l - 1} + kronecker(matrix(1,1,numSamples),stack{l}.b);
      y_hat = exp(y_hat);
      
      
      hAct{l} = bsxfun(@rdivide, y_hat, sum(y_hat, 1));
      [pred_prob pred_labels] = max(hAct{l});
      
      pred_prob = hAct{l};
      
      #       %% return here if only predictions desired.
      if po
          cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
          grad = [];
          return;
      end;
      

      
      ###########
      a{1} = data; % a(1)=X;
      for l = 2:nrLayers
          # z(l) = [W*a + b](l-1), for each sample,
          # where b is col.vector with same nr rows as W
          z = bsxfun(@plus, stack{l-1}.W*a{l-1}, stack{l-1}.b);
          if l == nrLayers && strcmp(ei.output_type, 'categorical')
            # If classifying into categories, then for last layer,
            # don't do full sigmoid, just exp, to calculate softmax below in pred_prob
            a{l} = exp(z);
          else
            # For other layers, or for continuous output, DO calculate sigmoid
            a{l} = f(z);
          end
      end
      
#       % IF categorical,
#       % normalize final layer, so each sample's outputs sum to 1,
#       % so that they are actual prediction probabilities.
#       % Final layer outputs are in a, with nrClasses rows and nrSamples cols,
#       % so we need something like (a ./ sum(a,1)), arranged right...
#       % but IF continuous, just use a{nrLayers} as the prediction.
      switch ei.output_type
        case 'categorical'
          pred_prob = bsxfun(@rdivide, a{nrLayers}, sum(a{nrLayers},1));
        case 'continuous'
          pred_prob = a{nrLayers};
      end

%% return here if only predictions desired.
if po
  cost = -1;
  grad = [];  
  return;
end;

      
      
      ######################################
      #       %% compute cost
      #       %%% YOUR CODE HERE %%%
      y_hat = log(hAct{numHidden+1});
      index = sub2ind(size(y_hat), t(labels), 1:numSamples);
      ceCost = -sum(y_hat(index));
      
      
      
      switch ei.output_type
          case 'categorical'
              # Find linear equivalents of matrix indices (i,j)
              # where, in j'th sample (column), i=y(j)
    	        # i.e. the row ID is the class of that column
              	IDs = sub2ind(size(pred_prob), t(labels), 1:nrSamples);
              # Compute binary matrix where IDs are 1 and rest are 0
              y_matrix = zeros(size(pred_prob));
              y_matrix(IDs) = 1;
              # Cost function f=J(theta), NOT yet including L2 penalty on the weights
              cost = -sum(log(pred_prob(IDs)));
          case 'continuous' 
            #use squared error loss, but DON'T divide by NrSamples
            cost = .5 .* sum(sum((pred_prob - labels).^2));
      end



      
      ######################################     
      #       %% compute gradients using backpropagation
      #       %%% YOUR CODE HERE %%%
      #         % Cross entroy gradient
      targets = zeros(size(hAct{numHidden+1})); # numLabels * numSamples
      targets(index) = 1;
      gradInput = hAct{numHidden+1} - targets;
      
      
      #for l =  numHidden+1 : -1 : 1
      for (l in seq(numHidden+1, 1, by = -1))
      {
          if (l > numHidden)
          {
            gradFunc = ones(size(gradInput));
          }
          else
          {
            #gradFunc = hAct{l} .* (1 - hAct{l}); ## derivative w.r.t logistic as well as softmax
            gradFunc = 1 - hAct{l} .^ 2; ## for tanh
          } 
            
          
          gradOutput = gradInput .* gradFunc;
          if (l > 1)
          {
              gradStack{l}.W = gradOutput * t(hAct{l-1});
          }
          else
          {
              gradStack{l}.W = gradOutput * t(data);
          }
    
          gradStack{l}.b = sum(gradOutput, 2);
          gradInput = t(stack{l}.W) * gradOutput;
      }
      
      ######################################
      #       %% compute weight penalty cost and gradient for non-bias terms

      
      wCost = 0;
      for (l in 1:numHidden+1)
      {
          wCost = wCost + .5 * ei.lambda * sum(stack{l}.W(:) .^ 2);
      }
      
      cost = ceCost + wCost;
      
      #       % Computing the gradient of the weight decay.
      for (l in seq(numHidden, 1, -1))
      {
          gradStack{l}.W = gradStack{l}.W + ei.lambda * stack{l}.W;
      }
      
      
      #       %% reshape gradients into vector
      [grad] = stack2params(gradStack);
}
      
