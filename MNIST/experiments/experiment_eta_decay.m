layers = 2;
neurons_h = 100;
momentum = [0,0.25,0.5,0.75,0.99];
eta = 0.001;
max_iter = 100;
momentum = 0.6;
randomize = 0;
w_init= 0;
batch_size = 1;
eta_decay = [1,25,50,75,100000000];
GPU = 0;

correct   = 0;
test_err  = zeros(2,100);
train_err = zeros(2,100);
runtime_gpu   = 0;
runtime_cpu   = 0;
iter      = 0;

for i = 1:length(eta_decay)
    disp(i)
    disp('//////////////////////////////////////')
    [correct, test_err(i,:), train_err(i,:), runtime_cpu, last_iter] = MLP(layers, neurons_h, eta , max_iter, momentum, randomize, w_init, batch_size,eta_decay(i), GPU);
    
end
%%
f=figure();
plot(train_err')
hold on
% plot(test_err')
xlabel('iteration')
ylabel('err')
legend(cellstr(num2str(eta_decay', '%-d')))

