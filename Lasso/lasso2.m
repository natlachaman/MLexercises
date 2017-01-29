clear all
input_train = dlmread('lasso_data/data1_input_train');
input_val = dlmread('lasso_data/data1_input_val');
output_train = dlmread('lasso_data/data1_output_train');
output_val = dlmread('lasso_data/data1_output_val');

max_iter = 10;
A = input_train';
% A = A(:,1:50);
b = output_train';
n = size(A,2);       % number of features
p = size(A,1);       % number of patterns
x = zeros(n,max_iter);
x(:,1) = 2*rand(n,1)-1;
gamma = 0.00:0.05:1;
final_beta = zeros(n,length(gamma));

% X0 = zeros(n,1);
% k  = 1;
% while (k<=max_iter - 1)
%     for i = 1:n
%         res = -A(i,[1:i-1,i+1:end])*X0([1:i-1,i+1:end]);
%         x(i) = (1/A(i,i)) * (res + b(i));
%     end
%     disp(norm(x-X0));
%     k = k+1;
%     X0 = x;
%     disp(immse(A*x,b))
%     
% end
% 
% break;
% for t = 2:max_iter
%     for i = 1:n
%         for u = 1:p
%             r = A(u,[1:i-1,i+1:end])*x([1:i-1,i+1:end],t-1);
%             x(i,t) = x(i,t) + (1/A(u,i)) * r + b(u)  ;
%         end
%         x(i,t-1) = 0.01* x(i,t)/p;
%     end
%     disp(immse(A*x(:,t),b))
% end
% 
% break;
% mse  = zeros(1,max_iter);
for g = 1:length(gamma)
    for k = 2:max_iter
        for i = 1:n
            tmpx = 0;
            for u = 1:p
                tmp = 0;
                for j = 1:n
                    if j~=i
                        tmp = tmp - A(u,j) * x(j,k-1);
                    end
                end
                r = A(u,[1:i-1,i+1:end])*x([1:i-1,i+1:end],t-1);
                tmpx = tmpx +  (tmp-b(u))*A(u,i);
            end
            tmpx = tmpx/p;   % weight the new estimate
            x(i,k) = sign(tmpx) * max(abs(tmpx)-gamma(g),0); % from slide 217
        end
        disp(immse(A*x(:,k),b))
    end
    final_beta(:,g) = x(:,max_iter);
end
gamma0 = repmat(final_beta(:,2),1,length(gamma));
% final_beta = final_beta./gamma0;
plot(gamma,final_beta')


