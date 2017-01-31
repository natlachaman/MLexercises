
function [beta, MSE] = lasso2(A,b, lambda,max_iter)
    n = size(A,2);       % number of features
    p = size(A,1);       % number of patterns
    MSE = zeros(length(lambda),1);
    final_beta = zeros(n,length(lambda));
    x = zeros(n,max_iter);
    for g = 1:length(lambda)
        for k = 2:max_iter
            for i = 1:n
                tmpx = 0;
                for u = 1:p
                    r = A(u,[1:i-1,i+1:end])*x([1:i-1,i+1:end],k);
                    tmpx = tmpx +  (b(u)-r)*A(u,i);
                end
                tmpx = tmpx/p;   % weight the new estimate
                x(i,k) = sign(tmpx) * max(abs(tmpx)-lambda(g),0); % from slide 217
            end
        
        end
        MSE(g) = immse(A*x(:,k),b);
        final_beta(:,g) = x(:,max_iter);
    end
%     gamma0 = repmat(final_beta(:,1),1,length(gamma));
%     final_beta = final_beta./gamma0;
    beta = final_beta';
    
end

