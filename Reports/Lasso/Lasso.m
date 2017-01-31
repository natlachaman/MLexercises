function Lasso
rng(18)
input_train = dlmread('lasso_data/data1_input_train');
input_val = dlmread('lasso_data/data1_input_val');
output_train = dlmread('lasso_data/data1_output_train');
output_val = dlmread('lasso_data/data1_output_val');
max_iter = 100;
A = input_train';
b = output_train';
beta_init = 2*rand(1,size(A,2))-1;

lambda = logspace(-2,0,25);
betas = zeros(100,length(lambda));
MSEs = zeros(1,length(lambda));

for i = 1: length(lambda)
     % create cv-splits
     disp(i)
     splits = 5;
     for cv = 1:splits
         cv_all = 1:size(A,1);
         cv_all = cv_all(randperm(size(A,1)));
         cvtest = cv_all(1:end/splits);
         cvtrain  = cv_all(end/splits:end);
         % get betas and mse on train set
         [cv_beta, cv_mse, cv_best_beta] = lasso2(A(cvtrain,:),b(cvtrain,:), lambda(i),max_iter,beta_init);
         % see how we perform on test set
         cv_val = immse(A(cvtest,:)*cv_best_beta,b(cvtest));
         MSEs(i) = MSEs(i) + cv_val;
     end
     MSEs(i) = MSEs(i) / splits;
     betas(:,i) = betas(:,i) / splits;
     
     [all_beta, all_mse, all_best_beta(:,i)] = lasso2(A, b, lambda(i), max_iter, beta_init);
     all_val(i) = immse(input_val'*all_best_beta(:,i),output_val');
end
%%  plot CV results
 best_lambda = lambda(all_val==min(all_val));
disp(['Best lambda seems to be: ', num2str(best_lambda)])
figure()
title('Lasso Regression')
subplot(1,2,1)
plot(all_best_beta(:,2:end)')
set(gca, 'XTick', [1:5:25])
set(gca, 'XTickLabel', lambda(1:5:end))
xlabel('Lambda')
ylabel('Coefficient magnitude')
title('Lasso Regression')

subplot(1,2,2)
plot(log(all_val))
hold on
plot(log(MSEs))
set(gca, 'XTick', [1:5:25])
set(gca, 'XTickLabel', lambda(1:5:end))
xlabel('Lambda')
ylabel('MSE on validation set (logscaled)')
[~,min_idx] = min(MSEs);
line([min_idx min_idx],get(gca,'YLim'),'Color',[1 0 0]);
legend('Validation-MSE','CV-MSE')

%% Ridge regression best parameter
rng(16)
lambda = logspace(0,5,25);
betas = zeros(100,length(lambda));
MSEs = zeros(1,length(lambda));
rng()
for i = 1: length(lambda)
     % create cv-splits
     disp(i)
     splits = 5;
     for cv = 1:splits
         cv_all = 1:size(A,1);
         cv_all = cv_all(randperm(size(A,1)));
         cvtest= cv_all(1:end/splits);
         cvtrain  = cv_all(end/splits:end);
         % get betas and mse on train set
         cv_best_beta = ridge(b(cvtrain,:),A(cvtrain,:), lambda(i));
         % see how we perform on test set
         cv_val = immse(A(cvtest,:)*cv_best_beta,b(cvtest));
         MSEs(i) = MSEs(i) + cv_val;
     end
     MSEs(i) = MSEs(i) ./ splits;
     betas(:,i) = betas(:,i) / splits;
     
     all_best_beta_ridge(:,i) = ridge(b, A, lambda(i));
     all_val(i) = immse(input_val'*all_best_beta_ridge(:,i),output_val');
end

%% Plot CV results for ridge

best_lambda_ridge = lambda(MSEs==min(MSEs));
[~, min_idx_ridge] = min(MSEs);
disp(['Best lambda seems to be: ', num2str(best_lambda_ridge)])
figure()

subplot(1,2,1)
plot(all_best_beta_ridge(:,2:end)')
set(gca, 'XTick', [1:5:25])
set(gca, 'XTickLabel', lambda(1:5:end))
xlabel('k')
ylabel('Coefficient magnitude')
title('Ridge Regression')

subplot(1,2,2)
plot(all_val)
hold on
plot(MSEs)
set(gca, 'XTick', [1:5:25])
set(gca, 'XTickLabel', lambda(1:5:end))
xlabel('k')
ylabel('MSE on validation set (logscaled)')
line([min_idx_ridge min_idx_ridge],get(gca,'YLim'),'Color',[1 0 0]);
legend('Validation-MSE','CV-MSE')


%% compare Ridge to Lasso

figure()
subplot(1,2,1)
bar(all_best_beta(:,min_idx))
xlabel('Lasso weights')
subplot(1,2,2)
bar(all_best_beta_ridge(:,min_idx))
xlabel('Ridge weights')

%% Correlated data

% example 1 from Zhao and Yu 2006
n=3;
p=1000;     % train data size
w=[2,3,0];  % example 1a
% w=[-2,3,0];    % example 1b
sigma=1;
x(1:2,:)=randn(2,p);
x(3,:)=2/3*x(1,:)+2/3*x(2,:)+1/3*randn(1,p);
y=w*x+randn(1,p);

max_iter=100;
A = x(:,1:700)';
b = y(1:700)';
A_val = x(:,701:end)';
b_val = y(701:end)';

b_corr = 1/size(x,2) * x*y'; % correlations

beta_init = 2*rand(1,size(A,2))-1;

lambda = logspace(-2,0,25);
betas = zeros(3,length(lambda));
MSEs = zeros(1,length(lambda));
%%
for i = 1: length(lambda)
     % create cv-splits
     disp(i)
     splits = 5;
     for cv = 1:splits
         cv_all = 1:size(A,1);
         cv_all = cv_all(randperm(size(A,1)));
         cvtest = cv_all(1:end/splits);
         cvtrain  = cv_all(end/splits:end);
         % get betas and mse on train set
         [cv_beta, cv_mse, cv_best_beta] = lasso2(A(cvtrain,:),b(cvtrain,:), lambda(i),max_iter,beta_init);
         % see how we perform on test set
         cv_val = immse(A(cvtest,:)*cv_best_beta,b(cvtest));
         MSEs(i) = MSEs(i) + cv_val;
     end
     MSEs(i) = MSEs(i) / splits;
     betas(:,i) = betas(:,i) / splits;
     
     [all_beta, all_mse, all_best_beta(:,i)] = lasso2(A, b, lambda(i), max_iter, beta_init);
     all_val(i) = immse(A_val*all_best_beta(:,i),b_val);
end
%%  plot CV results
 best_lambda = lambda(MSEs==min(MSEs));
disp(['Best lambda seems to be: ', num2str(best_lambda)])
figure()
title('Lasso Regression')
subplot(1,2,1)
plot(all_best_beta(:,2:end)')
set(gca, 'XTick', [1:5:25])
set(gca, 'XTickLabel', lambda(1:5:end))
xlabel('Lambda')
ylabel('Coefficient magnitude')
title('Lasso Regression')

subplot(1,2,2)
plot(log(all_val))
hold on
plot(log(MSEs))
set(gca, 'XTick', [1:5:25])
set(gca, 'XTickLabel', lambda(1:5:end))
xlabel('Lambda')
ylabel('MSE on validation set (logscaled)')
[~,min_idx] = min(MSEs);
line([min_idx min_idx],get(gca,'YLim'),'Color',[1 0 0]);
legend('Validation-MSE','CV-MSE')
disp(['Betas: ', mat2str(all_best_beta(:,min_idx))])
disp(['Real Betas: ', mat2str(b_corr)])

end


function [beta, MSE, best_beta] = lasso2(A, b, lambda, max_iter, beta_init)

    p = size(A,1);       % number of patterns
    n = size(A,2);       % number of features
    MSE = zeros(max_iter,1);
    x = zeros(n,max_iter);
    x(:,1) = beta_init;
    MSE(1) = immse(A*x(:,1),b);
    best_beta = beta_init;
%     x_hat = beta_init;
    
    for k = 1:max_iter
        for i = 1:n
            x_hat = 0;
            for u = 1:p
                r = A(u,[1:i-1,i+1:end])*x([1:i-1,i+1:end],k);  % sum over all products except j~=i
                x_hat = x_hat + (b(u)-r)*A(u,i);
            end
            x_hat = 1/p * x_hat;   % weight the new estimate
            x(i,k) =  sign(x_hat) * max(abs(x_hat)-lambda,0); % from slide 217
        end
        
        x(:,k+1) = x(:,k); % copy for new iteration
        MSE(k+1) = immse(A*x(:,k),b);
        if MSE(k+1) < MSE(k) 
            best_beta = x(:,k);
        end
        beta = x(:,1:max_iter);

        if k > 5 && norm(x(:,k)-x(:,k-1))/norm(x(:,k)) < 0.01 % early stopping criterium
            beta = x(:,1:max_iter);
            disp('converged earlier')
            return
        end
    end
end

