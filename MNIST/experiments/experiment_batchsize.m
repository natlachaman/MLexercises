layers = 2;
neurons_h = 100;
eta = 0.00001;
max_iter = 100;
momentum = 0;
randomize = 1;
w_init= 0;
eta_decay = 0;
GPU = 0;
batch_size = [1,16,64,128,1024,4096,11791];

correct   = 0;
test_err  = zeros(length(batch_size),100);
train_err = zeros(length(batch_size),100);
runtime_gpu   = 0;
runtime_cpu   = 0;
iter          = 0;

for i = 1:length(batch_size)
    disp(i)
    disp('//////////////////////////////////////')
    [correct, test_err(i,:), train_err(i,:), runtime_cpu, last_iter] = MLP(layers, neurons_h, eta , max_iter, momentum, randomize, w_init, batch_size(i),eta_decay, GPU);
end
%%
f=figure();
plot(train_err')
hold on
% plot(test_err')
xlabel('iteration')
ylabel('err')
legend(cellstr(num2str(batch_size', '%-d')))

