layers = [2,3,4];
neurons = [1,2,5,10,50,100,200,300,500,784,1000,2000,5000];
momentum = [0,0.25,0.5,0.75,0.99];
max_iter = 500;

correct   = 0;
test_err  = 0;zeros(length(momentum),100);
train_err = 0;zeros(length(momentum),100);
runtime_gpu   = 0;
runtime_cpu   = 0;
iter      = 0;

% for i = 1:length(momentum)
%     disp(i)
%     disp('//////////////////////////////////////')
    [correct, test_err, train_err, runtime_cpu, last_iter] = MLP(2, 0.001, 11791, max_iter, 0.75, 1);;
% end
%%
f=figure();
plot(train_err')
hold on
plot(test_err')
xlabel('iteration')
ylabel('err')
legend('train_err','test_err')

