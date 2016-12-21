% correct = mlp_predict(w_in,w_h,w_out, test, labels)
function [correct test_err] = mlp_predict_GPU(w, bias, test, labels)
    layers      = length(w);
    neurons_h   = length(w{1});
    neurons_out = size(w{end}, 1);
    x           = cell(1,layers);
    test_err    = 0;
    
    for k = 1:layers-1
       x{k}  = gpuArray(zeros(neurons_h,1));    % what comes out of each layer
    end
    x{end}  = gpuArray(zeros(neurons_out,1));         % final output neuron(s)
    correct = gpuArray(zeros(length(test),1));
    for u = 1:length(test)
        % forward step
        tic
        img = test(:,u);           % get input image
        x{1}  = tanh(w{1} * img + bias);             % calulate output for first layer
        for k = 2:layers
            x{k} = tanh( w{k}* x{k-1} + bias);       % calculate output of that hidden layer
        end   
        err = 0.5 * (labels(u) - gather(x{end})).^2;
        test_err = test_err + sum(err);
        correct(u) = sign(gather(x{end}))==labels(u,:)    ;              % was our prediction correct?
    end
    test_err = gather(test_err);
    correct  = gather(100*sum(correct)/length(test));
end