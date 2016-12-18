% cls
close all
load('mnistAll.mat')
rng(42)
% define parameters
eta = 0.00001;                 % learning rate
layers = 2;                  % # hidden layers
neurons_h = 784;             % # neurons per hidden layer
neurons_in = 784;            % # input neurons
neurons_out = 1;             % # output neurons
max_iter = 10500;             % # iterate for so long
bias = -1;
assert(layers>0);   % layers must be at least 1

% define weights matrixes
w.in  = gpuArray(rand(neurons_h,neurons_in)*2-1);
w.out = gpuArray(rand(neurons_out,neurons_h)*2-1);
w.h   = gpuArray(rand(neurons_h,neurons_h,layers-1)*2-1);

% define input and output memory for each neuron
y.h_in   = gpuArray(zeros(neurons_h,layers));    % what comes in to each layer
y.h_out  = gpuArray(zeros(neurons_h,layers));    % what comes out of each layer
y.o_in   = gpuArray(zeros(neurons_out));         % final output neuron(s)
y.o_out  = gpuArray(zeros(neurons_out));         % final output neuron(s)

% define delta values for each neuron
d.out = gpuArray(zeros(neurons_out));
d.h   = gpuArray(zeros(neurons_h,layers));


% get train/test data
train = double(mnist.train_images(:,:,(mnist.train_labels==1) | (mnist.train_labels==8)));
train_label = double(mnist.train_labels((mnist.train_labels==1) | (mnist.train_labels==8)));
train_label(train_label==1) = -1;
train_label(train_label==8) = 1;
t = train_label;
test = double(mnist.test_images(:,:,(mnist.test_labels==1) | (mnist.test_labels==8)));
test_label = double(mnist.test_labels((mnist.test_labels==1) | (mnist.test_labels==8)));
test_label(test_label==1) = -1;
test_label(test_label==8) = 1;

% normalize (makes it more efficient)
train = (train)*2  / 255 -1;
test  = (test)*2   / 255 -1;

errors = [];
converged = false;
iter = 0;
tic
disp(['Starting at ', datestr(rem(now,1))])

while(~converged && iter ~= max_iter)
    iter = iter + 1;
    total_err = 0;
    for u = 1:length(train)/100
        % forward step
        img = reshape(train(:,:,u),1,784)';                % get input image
        y.h_out(:,1)  = tanh(w.in * img + bias);                 % calulate output for first layer
        for h = 1:layers-1
            y.h_out(:,h+1) = tanh(w.h(:,:,h)* y.h_out(:,h) + bias);         % calculate output of that hidden layer
        end
        y.o_out   =  tanh(w.out * y.h_out(:,end) + bias);                % calulate final output
        err = 0.5 * (train_label(u) - y.o_out).^2;
        total_err = total_err +  sum(err);

        % backpropagation
        d.out = (y.o_out-t(u)) .* (1-tanh(y.o_in).^2);     % calculate delta for output neuron ERROR/dx * ACTIVATION/dx 
        w.out = w.out - (eta * d.out * y.h_out(:,end)');
        
        
        for h = layers-1:-1:1
              if h == layers - 1
                 d.h(:,h) = (w.out' * d.out) .*(1-tanh(y.h_out(:,h)).^2);
              else
                 d.h(:,h) = (w.h(:,:,h)' * d.h(:,h+1)) .*(1-tanh(y.h_out(:,h)).^2);
              end
              w.h(:,:,h) = w.h(:,:,h) - eta * d.h(:,h) *  y.h_out(:,h)';
        end
        
        if layers == 1
            d.in = (w.out' * d.out) .* (1-tanh(img).^2);
        else
            d.in =     w.in' * d.h(:,1) .* (1-tanh(img).^2);
        end
        
        w.in = w.in - eta * d.in * img';

    end
    errors(iter) = gather(total_err);
    if iter > 10 && var(errors(end-4:end)) < eta*eta; converged = 1; end
    disp(['iteration ', num2str(iter), ', Err: ', num2str(total_err)])%, ', Predict: ' , num2str(mlp_predict(w.in,w.h,w.out, test, test_label))])
    if mod(iter,10) == 0; disp([', Predict: ' , num2str(mlp_predict(w.in,w.h,w.out, test, test_label))]);end
end

endtime = toc;
% corr = mlp_predict(w.in, w.h, w.out, test, test_label);
disp(['Total correct: ', sprintf('%.2f',corr) , '%, Runtime: ', sprintf('%.3f',endtime/60),' min'])


% save current workspace w/o mnist
name = [num2str(layers), '-',num2str(neurons_h),'-', num2str(eta),' - ',num2str(iter),' - ', num2str(errors(end)),' - ',sprintf('%.2f',corr),'.mat'];
save( name, '-regexp','^(?!(test_label|test|train|train_label|mnist|t)$).')
% disp('end')