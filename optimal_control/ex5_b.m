

n=200;
J=[];
lambda=1;
for x=-2:0.1:2
    for v=-2:0.1:2
        E=0;
        E_all=[];
        for sample=1:n % perform Metropolis Hasting step
            E_new=FinalEnergy(x,v,lambda); % calculate the -phi(T)/lambda
            Delta=-(E_new-E);
            if (Delta < 0)
                E=E_new;
            else
                if (rand< exp(-Delta))
                    E=E_new;
                end;
            end;
            E_all=[E_all E];
        end;
        [x,v]
        J=[J; [x,v,-lambda*log(mean(exp(E_all)))]]; % append the new datapoint for J
    end;
end;

X = reshape(J(:,1),[41,41]);
V = reshape(J(:,2),[41,41]);
JJ = reshape(J(:,3),[41,41]);
surf(X,V,JJ)
xlabel('x')
ylabel('v')
zlabel('J')