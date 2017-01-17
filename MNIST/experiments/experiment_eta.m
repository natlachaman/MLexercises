layers = [2,3,4];
neurons = [1,2,5,10,50,100,200,300,500,784,1000,2000,5000];
momentum = [0,0.25,0.5,0.75,0.99];
batch_size = [1,2,4,8,16,32,64,128];
max_iter = 500;
eta = [0.000001,0.0001,0.0005,0.001,0.005,0.01,0.1,1];

correct   = 0;
test_err  = zeros(length(eta),100);
train_err = zeros(length(eta),100);
runtime_gpu   = 0;
runtime_cpu   = 0;
iter      = 0;

for i = 1:length(eta)
    disp(i)
    disp('//////////////////////////////////////')
    [correct, test_err(i,:), train_err(i,:), runtime_cpu, last_iter] = MLP(2, eta(i), 100, 100,0,1,  0.6, 0);
end
%%
f=figure();
plot(train_err')
hold on
% plot(test_err')
xlabel('iteration')
ylabel('err')
legend('0.000001','0.00001','0.0001','0.001','0.01','0.1','0.5','1','10')

