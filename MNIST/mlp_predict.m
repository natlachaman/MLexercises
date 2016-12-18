% correct = mlp_predict(w_in,w_h,w_out, test, labels)
function correct = mlp_predict(w_in,w_h,w_out, test, labels)

    layers = size(w_h,3);
    neurons_h   = length(w_h);
    neurons_out = size(w_out, 1);

    y.h_in   = gpuArray(zeros(neurons_h,layers));    % what comes in to each layer
    y.h_out  = gpuArray(zeros(neurons_h,layers));    % what comes out of each layer
    y.o_in   = gpuArray(zeros(neurons_out));         % final output neuron(s)
    y.o_out  = gpuArray(zeros(neurons_out));         % final output neuron(s)
    correct  = gpuArray(zeros(1,length(test)));
    bias     = -1;
    
    for u = 1:length(test)
        
        img = reshape(test(:,:,u),1,784)';                % get input image
        y.h_in(:,1)   = w_in * img;                       % calculate input of first layer
        y.h_out(:,1)  = tanh(y.h_in(:,1) + bias);         % calulate output for first layer
        
        for h = 1:layers-1
            y.h_in(:,h+1)  = w_h(:,:,h)* y.h_out(:,h);    % calculate input for layer h+1
            y.h_out(:,h+1) = tanh(y.h_in(:,h+1) + bias);  % calculate output of that hidden layer
        end
        
        y.o_in    =  w_out * y.h_out(:,end);
        y.o_out   =  tanh(y.o_in + bias);                 % calulate final output
        correct(u) = sign(y.o_out)==labels(u,:)    ;              % was our prediction correct?
    end
    correct = 100*sum(correct)/length(test);
end