% cls
% function [correct, test_err, train_err, runtime, last_iter] = MLP(layers, eta, neurons_h, max_iter, momentum, GPU)

load('mnistAll.mat')
rng(18)
% settings
GPU = 0;
momentum = 0.0;

% define parameters
eta = 0.001;                 % learning rate
layers = 3;                  % # layers  =(hidden layers + 1)
neurons_h = 5;              % # neurons per hidden layer
neurons_in = 784;              % # input neurons
neurons_out = 1;               % # output neurons
max_iter = 100;            % # iterate for so long
bias = -1;
assert(layers>1);            % layers must be at least 2
class_1   = 4;
class_2   = 9;
randomize = 0;

% define weights matrixes
w = cell(1,layers);
w{1}  = rand(neurons_h,neurons_in)*2-1;
for k = 2:layers
   w{k}  = rand(neurons_h,neurons_h)*2-1;
end
w{end} = rand(neurons_out,neurons_h)*2-1;


% define input and output memorx for each neuron
x        = cell(1,layers);
for k = 1:layers-1
   x{k}  = zeros(neurons_h,1);    % what comes out of each layer
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
m{1}  = ones(neurons_h,neurons_in);
for k = 2:layers
   m{k}  = ones(neurons_h,neurons_h);
end
m{end} = ones(neurons_out,neurons_h);

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

% normalize (makes it more efficient)
train = (train)*2  / 255 -1;
test  = (test)*2   / 255 -1;
% train_label = gpuArray(train_label);
% test_label  = gpuArray(test_label);

train = reshape(train,784,length(train));
test  = reshape(test,784,length(test));

errors  = [];
correct = [];
test_err = [];
converged = 0;
iter = 0;

disp(['Starting at ', datestr(rem(now,1))])
% transform rest to GPU
bias = bias;
% train_label = gpuArray(train_label);
t = t;

if GPU
    disp('Using GPU...')
    bias=gpuArray(bias); t=gpuArray(t); 
    test=gpuArray(test); train=gpuArray(train);
    for k = 1:layers
        w{k}=gpuArray(w{k}); 
        d{k}=gpuArray(d{k}); 
        x{k}=gpuArray(x{k}); 
    end
else
    disp('Using CPU...');
end




%print initial performance
mlp_predict(w, bias, test, test_label)
eta_0 = eta;
tic
t_start = tic;
while(~converged && iter ~= max_iter)
    iter = iter + 1;
    total_err = 0;
    eta = eta_0 /(1 + iter/50);
%     tic 
    starttime = tic;
    acc = 0;
	if randomize==1
		ordering = randperm(length(train));
		train = train(:, ordering);
		train_label = train_label(ordering);
    end
    
%     forward step

    for u = 1:length(train)
        
        img = train(:,u);          % get input image
        x{1}  = tanh(w{1} * img + bias);             % calulate output for first layer
        for k = 2:layers
            x{k} = tanh( w{k}* x{k-1} + bias);       % calculate output of that hidden layer
        end
        acc = acc + sum(sign(gather(x{end}))==train_label(u,:));
        
        err = 0.5 * (train_label(u) - gather(x{end})).^2;
        total_err = total_err +  sum(err);
    
%     backpropagation

        for k=layers:-1:1;

            if k == layers;
                o = t(u);
                i = x{k-1};
                mat = x{k};
            elseif k == 1;
                o = d{k+1};
                i = img;
                mat = w{k+1}';
            else
                o = d{k+1};
                i = x{k-1};
                mat = w{k+1}';
            end

            d{k} = mat * o .* (1-tanh(w{k}*i).^2);
            m{k} = m{k} + (d{k}*i');
        end
        
    end
    
    
    % conjugate gradient update
    for k=layers:-1:1;
        
        if iter == 1;
            dir{k} = - m{k};
            m_{k} = m{k};
        else
            dir_{k} = dir{k};
%             beta = ((m{k} - m_{k}) * m{k}')./(m_{k}*m_{k}');
%             beta = beta(logical(eye(length(beta))));
%             beta = repmat(beta,1,size(m{k},2));
            beta = ((m{k} - m_{k}) .* m{k})./(m_{k}.* m_{k});
            beta = sum(sum(beta));
            
            dir{k} = - m{k} + beta * dir_{k};
            m_{k} = m{k};
        end
        
        w{k} = w{k} + 0.00001 * dir{k};
        
    end
    
    acc = acc / length(train);
    errors(iter)   = gather(total_err);
    [correct(iter), test_err(iter)]  = mlp_predict(w, bias, test, test_label);
   
    if iter > 10
        if sum( test_err(end-5:end)<mean(test_err(end-10:end-5))) == 0
%             converged = 1;
        end
    end
    
    disp(['#', num2str(iter), ', Err: ', num2str(errors(iter)/length(train)),', Test-Err: ',num2str(test_err(iter)/length(test)), ', Predict: ' , num2str(correct(iter)), ' Tr-acc:' ,num2str( acc) ,', Time: ', num2str(int32(toc(starttime))),'s'])

end
endtime = toc;
disp([sprintf('%.2f',endtime/60),' min'])


% save current workspace w/o mnist
name = [num2str(layers), '-',num2str(neurons_h),'-', num2str(eta),' - ',num2str(iter),' - ', num2str(errors(end)),' - ',num2str(test_err(iter)), ' - ',sprintf('%.2f',max(correct)),'.mat'];
save( name, '-regexp','^(?!(test_label|test|train|train_label|mnist|t|d|x|img)$).')
% disp('end')
disp(['Finished at ', datestr(rem(now,1))])
runtime   = toc(t_start);
train_err = errors;
correct   = correct;
last_iter =  iter;
% end