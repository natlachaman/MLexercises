layers = 2;
neurons_h = 100;
momentum = [0,0.25,0.5,0.75,0.99];
eta = 0.0001;
max_iter = 100;
momentum = 0.6;
randomize = 0;
w_init= 0;
batch_size = 1;
eta_decay = 0;
GPU = 0;

correct   = 0;
test_err  = zeros(2,max_iter);
train_err = zeros(2,max_iter);
runtime_gpu   = 0;
runtime_cpu   = 0;
iter      = 0;

% for i = 1:length(batch_size)
    disp(i)
    disp('//////////////////////////////////////')
    [correct, test_err(1,:), train_err(1,:), runtime_cpu, last_iter] = MLP(2, 100, eta , max_iter, momentum, randomize, w_init, batch_size,eta_decay, GPU);
    [correct, test_err(2,:), train_err(2,:), runtime_cpu, last_iter] = MLP(5, 20, eta , max_iter, momentum, randomize, w_init, batch_size,eta_decay, GPU);
end
%%
f=figure();
plot(train_err(2,:)' / 11791)
hold on
plot(test_err(2,:)' / 1991)
xlabel('iteration')
ylabel('err')
title('Deep network')
legend('train-err','test-err' )