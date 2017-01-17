% cls
function [correct, test_err, train_err, runtime, last_iter] = MLP(layers, neurons_h, eta , max_iter, momentum, randomize, w_init, batch_size,eta_decay, GPU)

load('mnistAll.mat')
rng(18)
% settings
if ~exist('GPU','var') || isempty(GPU)
  GPU = 0;  % use GPU or not
end
if ~exist('momentum','var') || isempty(momentum)
  momentum = 0; % Used momentum beta-value
end
if ~exist('eta','var') || isempty(eta)
  eta = 0.001;  % learning rate
end
if ~exist('layers','var') || isempty(layers)
  layers = 2;   % # layers + output layer (hidden layer + 1)
end
if ~exist('neurons_h','var') || isempty(neurons_h)
  neurons_h = 100;      % # neurons per hidden layer
end
if ~exist('max_iter','var') || isempty(max_iter)
  max_iter = 100;   % maximum iterations to run
end
if ~exist('randomize','var') || isempty(randomize)
  randomize = 0;    % randomize before each epoch?
end
if ~exist('w_init','var') || isempty(w_init) || w_init==0
  w_init = 1/sqrt(neurons_h);   % init our weights to what scale
end
if ~exist('batch_size','var') || isempty(batch_size)
  batch_size = 1;       % how many samples per mini batch. 1 = online learning
end
if ~exist('eta_decay','var') || isempty(eta_decay)|| eta_decay==0
  eta_decay = inf;  % inf = no learning rate decay
end

% define parameters
neurons_in = 784;              % # input neurons
neurons_out = 1;               % # output neurons
bias = -1;                     % specify the bias
assert(layers>1);              % layers must be at least 2
class_1   = 4;          % use these two numbers
class_2   = 9;

sprintf(['Settings\nLayers: ',num2str(layers), '\nNeurons: ',num2str(neurons_h),...
    '\nLearning-rate: ',num2str(eta), '\nIterations: ',num2str(max_iter),...
    '\nMomentum: ', num2str(momentum), '\nRandomize: ',num2str(randomize),...
    '\nW_init: ', num2str(w_init), '\nBatch-size: ', num2str(batch_size)
    ])

% define weights matrixes
w = cell(1,layers);
w{1}  = (rand(neurons_h,neurons_in)*2-1) * w_init;
for k = 2:layers
   w{k}  = (rand(neurons_h,neurons_h)*2-1) * w_init;
end
w{end} = (rand(neurons_out,neurons_h)*2-1) * w_init;


% define input and output memorx for each neuron
x        = cell(1,layers);
for k = 1:layers-1
   x{k}  = zeros(neurons_h,1);          % what comes out of each layer
end
x{end}  = zeros(neurons_out,1);         % final output neuron(s)

% define delta
d = cell(1,layers);
for k = 1:layers-1
    d{k}  = zeros(neurons_h,1);
end
d{end}  = zeros(neurons_out,1);

% define gradient memory
m = cell(1,layers);
m{1}  = zeros(neurons_h,neurons_in);
for k = 2:layers
   m{k}  = zeros(neurons_h,neurons_h);
end
m{end} = zeros(neurons_out,neurons_h);

% get train/test data
train = double(mnist.train_images(:,:,(mnist.train_labels==class_1) | (mnist.train_labels==class_2)));
train_label = double(mnist.train_labels((mnist.train_labels==class_1) | (mnist.train_labels==class_2)));
train_label(train_label==class_1) = -1;
train_label(train_label==class_2) = 1;
t = train_label;
test = double(mnist.test_images(:,:,(mnist.test_labels==class_1) | (mnist.test_labels==class_2)));
test_label = double(mnist.test_labels((mnist.test_labels==class_1) | (mnist.test_labels==class_2)));
test_label(test_label==class_1) = -1;
test_label(test_label==class_2) = 1;

% normalize (makes it more efficient) and reshape to array
train = (train)*2  / 255 -1;
test  = (test)*2   / 255 -1;
train = reshape(train,784,length(train));
test  = reshape(test,784,length(test));

% initialize variables
errors  = [];
correct = [];
test_err = [];
converged = 0;
iter = 0;

disp(['Starting at ', datestr(rem(now,1))])

if GPU  % if GPU is used, push arrays to GPU
    disp('Using GPU...')
    bias=gpuArray(bias); t=gpuArray(t); 
    test=gpuArray(test); train=gpuArray(train);
    for k = 1:layers
        w{k}=gpuArray(w{k}); 
        d{k}=gpuArray(d{k}); 
        x{k}=gpuArray(x{k}); 
        m{k}=gpuArray(m{k});
    end
else
    disp('Using CPU...');
end




%print initial performance
disp(['Initial correct on test-set: ',num2str(predict(w, bias, test, test_label))])
eta_0 = eta;
tic
t_start = tic;

while(~converged && iter ~= max_iter)
    iter = iter + 1;
    total_err = 0;
    eta = eta_0 /(1 + iter/eta_decay);
    starttime = tic;
    acc = 0;
    
	if randomize==1     % shuffle train set
		ordering = randperm(length(train));
    else
        ordering = 1:length(train);
    end
    
    % begin new epoch
    for sample = 1:length(train)
        
        % forward step
        u = ordering(sample);
        img = train(:,u);                            % get input image
        x{1}  = tanh(w{1} * img + bias);             % calulate output for first layer
        for k = 2:layers
            x{k} = tanh( w{k}* x{k-1} + bias);       % calculate output of that hidden layer
        end
        acc = acc + sum(sign(gather(x{end}))==train_label(u,:));    % add 1 if correct, 0 if not correct class
        
        err = 0.5 * (train_label(u) - gather(x{end})).^2;   % calculate error of last layer
        total_err = total_err +  sum(err);
        
%         backpropagation
        d{end} = (x{end}-t(u)) .* (1-tanh(w{end}*x{end-1}).^2) ;  
        m{end} =  ( d{end} * x{end-1}') + m{end} * momentum + m{end};
        if mod(u, batch_size) == 0  % push changes every mini-batch size
            w{end}  =  w{end} - eta * m{end};
            m{end} = zeros(neurons_out,neurons_h);
        end
        
        for k = layers-1:-1:2           % gradients for hidden layers
            d{k} = w{k+1}' * d{k+1} .* (1-tanh(w{k}*x{k-1}).^2) ;
            m{k} =  d{k}*x{k-1}' + m{k} * momentum;
            if mod(u, batch_size) == 0  % push changes every mini-batch size
               w{k} = w{k} - eta * m{k};
               m{k} = zeros(neurons_h,neurons_h);
            end
        end

        d{1} = w{2}' * d{2} .* (1-tanh(w{1}*img).^2) ;  % gradients for input weights
        m{1} =  d{1}*img' + m{1} * momentum;
        if mod(u, batch_size) == 0  % push changes every mini-batch size
            w{1} = w{1} - eta * m{1};
            m{1} = zeros(neurons_h,neurons_in);
        end

    end
    
    acc = acc / length(train);  % calculate final accuracy
    errors(iter)   = gather(total_err); % save errors
    [correct(iter), test_err(iter)]  = predict(w, bias, test, test_label);
   
    if iter > 10        % possible convergence criterium (not used)
        if sum( test_err(end-5:end)<mean(test_err(end-10:end-5))) == 0
%             converged = 1;
        end
    end
    
    disp(['#', num2str(iter), ', Err: ', num2str(errors(iter)/length(train)),', Test-Err: ',num2str(test_err(iter)/length(test)), ', Predict: ' , num2str(correct(iter)), ' Tr-acc:' ,num2str( acc) ,', Time: ', num2str(int32(toc(starttime))),'s'])

end
endtime = toc;
disp([sprintf('%.2f',endtime/60),' min'])

% plot results
plot(errors'/length(train))
hold on
plot(test_err'/length(test))
legend('Train-err','Test-err' )
xlabel('Iteration')
ylabel('Error')
figure()
plot(correct')
xlabel('Iteration')
ylabel('Error')
legend('Accuracy')

% save current workspace w/o mnist
name = [num2str(layers), '-',num2str(neurons_h),'-', num2str(eta),' - ',num2str(iter),' - ', num2str(errors(end)),' - ',num2str(test_err(iter)), ' - ',sprintf('%.2f',max(correct)),'.mat'];
save( name, '-regexp','^(?!(test_label|test|train|train_label|mnist|t|d|x|img)$).')

disp(['Finished at ', datestr(rem(now,1))])
runtime   = toc(t_start);
train_err = errors;
last_iter =  iter;
end

function [correct, test_err] = predict(w, bias, test, labels)
    layers      = length(w);
    neurons_h   = length(w{1});
    neurons_out = size(w{end}, 1);
    x           = cell(1,layers);
    test_err    = 0;
    
    for k = 1:layers-1
       x{k}  = zeros(neurons_h,1,class(bias));    % what comes out of each layer
    end
    x{end}  = zeros(neurons_out,1,class(bias));         % final output neuron(s)
    correct = zeros(length(test),1,class(bias));
    
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