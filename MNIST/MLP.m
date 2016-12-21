% cls
close all
load('mnistAll.mat')
rng(40)
% define parameters
eta = 0.0005;                % learning rate
layers = 4;                  % # layers  =(hidden layers + 1)
neurons_h = 784;             % # neurons per hidden layer
neurons_in = 784;            % # input neurons
neurons_out = 1;             % # output neurons
max_iter = 100;             % # iterate for so long
bias = 0;
assert(layers>0);            % layers must be at least 1

% define weights matrixes
w = cell(1,layers);
w{1}  = rand(neurons_h,neurons_in)*2-1;
for l = 2:layers
   w{l}  = rand(neurons_h,neurons_h)*2-1;
end
w{end} = rand(neurons_out,neurons_h)*2-1;


% define input and output memorx for each neuron
x        = cell(1,layers);
for l = 1:layers-1
   x{l}  = zeros(neurons_h,1);    % what comes out of each layer
end
x{end}  = zeros(neurons_out,1);         % final output neuron(s)

% define delta
d = cell(1,layers);
for l = 1:layers-1
    d{l}  = zeros(neurons_h,1);
end
d{end}  = zeros(neurons_out,1);

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

errors  = [];
correct = [];
test_err = [];
converged = false;
iter = 0;
tic
disp(['Starting at ', datestr(rem(now,1))])

while(~converged && iter ~= max_iter)
    iter = iter + 1;
    total_err = 0;
%     tic 
    starttime = tic;
    for u = 1:length(train)
        % forward step
        img = reshape(train(:,:,u),1,784)';          % get input image
        x{1}  = tanh(w{1} * img + bias);             % calulate output for first layer
        for l = 2:layers
            x{l} = tanh( w{l}* x{l-1} + bias);       % calculate output of that hidden layer
        end
        err = 0.5 * (train_label(u) - x{end}).^2;
        total_err = total_err +  sum(err);

        % backpropagation
        d{end} = (x{end}-t(u)) .* (1-tanh(w{end}*x{end-1}).^2);  
        w{end}  = w{end} - (eta * d{end} * x{end}');
        
        for l = layers-1:-1:2
            d{l} = w{l+1}' * d{l+1} .* (1-tanh(w{l}*x{l-1}).^2);
            w{l} = w{l} - eta * d{l}*x{l-1}';
        end
        
        d{1} = w{2}' * d{2} .* (1-tanh(w{l}*img).^2);
        w{1} = w{l} - eta * d{1}*img';
%         if layers == 1
%            
%         else
%             d.in =     w.in' * d.h(:,1) .* (1-tanh(img).^2);
%         end
    end
    errors(iter)   = total_err;
    [correct(iter), test_err(iter)]  = mlp_predict(w, bias, test, test_label);
    
    disp(['#', num2str(iter), ', Err: ', num2str(errors(iter)/length(train)),', Test-Err: ',num2str(test_err(iter)/length(test)), ', Predict: ' , num2str(correct(iter)), ', Time: ', num2str(int32(toc(starttime))),'s'])

end

endtime = toc;
disp([sprintf('%.2f',endtime/60),' min'])


% save current workspace w/o mnist
name = [num2str(layers), '-',num2str(neurons_h),'-', num2str(eta),' - ',num2str(iter),' - ', num2str(errors(end)),' - ',num2str(test_err(iter)), ' - ',sprintf('%.2f',max(correct)),'.mat'];
save( name, '-regexp','^(?!(test_label|test|train|train_label|mnist|t|d|x|img)$).')
% disp('end')