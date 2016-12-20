% problem definition
% minimize E= 0.5 x^T w x, with x=x_1,...,x_n and x_i=0,1
% w is a symmetric real n x n matrix with zero diagonal

METHOD='sa';
NEIGHBORHOODSIZE=1;
n_restart =500;

tx = zeros(1,n_restart);
switch METHOD,
case 'iter'
    disp('Iterative Method')
	E_min = 1000;
	E_min_all = [];
	for t=1:n_restart,
	    tic
		% initialize
		x = 2*(rand(1,n)>0.5)-1;
		E1 = E(x,w);
		flag = 1;
		E_all = [E1];

		while flag == 1,
			switch NEIGHBORHOODSIZE,
			case 1,
				% choose new x by flipping one bit i
				% compute dE directly instead of subtracting E's of
				% different states because of efficiency
                A = x * w; % Taking it outside the loop
				for i=1:n
                    xN=x;
                    xN(i) = -1 * xN(i);
                    Delta = - (xN(i) - x(i)) * A(i);
%                     Delta = E(xN,w) - E1;
                    if (Delta<0)
                        E1 = E1 + Delta;
                        x=xN;
                        E_all = [E_all E1];
                        break;
                    end;
                end;
                % Terminate only if no new minimums are not found in the neighbourhood.
                if (i==n)
                    flag = 0;
                    tx(t) = toc;
                end;

			case 2,
				% choose new x by flipping bits i,j
				innerFlag = 1;
				A = x * w;
				for i=1:n
				    for j=1:n
                        xN=x;
                        xN(i) = -1 * xN(i);
                        if (j~=i)
                            xN(j) = -1 * xN(j);
                            a = xN(i)-x(i);
                            b = xN(j)-x(j);
%                             Delta = - (a*A(:,i)+b*A(:,j)) - 0.5 * (a*(a*w(i,i)+b*w(j,i))+b*(a*w(i,j)+b*w(j,j)));
                            Delta = - (a*A(:,i)+b*A(:,j)) - 0.5 * (a*b*w(j,i)+b*a*w(i,j));
                        else
                            Delta = - (xN(i) - x(i)) * A(i);
                        end;
%                         Delta = E(xN,w) - E1;
                        if (Delta<0)
                            E1 = E1 + Delta;
                            x=xN;
                            innerFlag=0;
                            E_all = [E_all E1];
                            break;
                        end;
                    end;
                    if (innerFlag==0) break; end;
                end;
                if (i==n && j==n) flag = 0; tx(t) = toc; end;
			end;
		end;
		if (E_min == E1) break;end;
		E_min = min(E_min,E1);
		E_min_all = [E_min_all E_min];
	end;
	E_min
	t
    plot(1:length(E_all),E_all)
	xlabel('Iteration')
	ylabel('Min Energy')

% 	plot(E_min_all)
% 	xlabel('Restart')
% 	ylabel('Min Energy')
% 	plot(tx)
% 	xlabel('Iterations')
% 	ylabel('Termination Time')


case 'sa'
	% initialize
    disp('Simulated Anealing')
	x = 2*(rand(1,n)>0.5)-1;
	E1 = E(x,w);
	E_outer=zeros(1,100);	%stores mean energy at each temperature
	E_bar=zeros(1,100);		% stores std energy at each temperature
    accept_all=[];
	% initialize temperature
	max_dE=0;
	switch NEIGHBORHOODSIZE,
        case 1,
			% estimate maximum dE in single spin flip
			A = x * w;
            for i=1:n
                xN=x;
                xN(i) = -1 * xN(i);
                Delta = - (xN(i) - x(i)) * A(i);
                max_dE = max(max_dE,Delta);
            end;
        case 2,
			% estimate maximum dE in pair spin flip
            A = x * w;
            for i=1:n
                for j=1:n
                    xN=x;
                    xN(i) = -1 * xN(i);
                    if (j~=i)
                        xN(j) = -1 * xN(j);
                        a = xN(i)-x(i);
                        b = xN(j)-x(j);
                        Delta = - (a*A(:,i)+b*A(:,j)) - 0.5 * (a*b*w(j,i)+b*a*w(i,j));
                    else
                        Delta = - (xN(i) - x(i)) * A(i);
                    end;
                    max_dE = max(max_dE,Delta);
                end;
            end;
        end;

	beta_init=1/max_dE;	% sets initial temperature
	T1=1000; % length markov chain at fixed temperature
	factor=1.05 ; % increment of beta at each new chain

	beta=beta_init;
	E_bar(1)=1;
	t2=1;
	Beta_all=[beta];
	HeatCapacity_all=[];
	while E_bar(t2) > 0,
	    C=beta^2 * E_bar(t2);
	    HeatCapacity_all=[HeatCapacity_all C];
		t2=t2+1;
		beta=beta*factor;
		Beta_all=[Beta_all beta];
		E_all=zeros(1,T1);
		for t1=1:T1,
			switch NEIGHBORHOODSIZE,
			case 1,
				% choose new x by flipping one random bit i
				% perform Metropolis Hasting step
				i = randi([1 n],1,1);
                x_new = x;
                x_new(i) = -1 * x_new(i);
                Delta = - (x_new(i) - x(i)) * x * w(:,i);
                delta=beta*Delta;
                if (delta < 0)
                    x=x_new;                   %accept x_new
                    E1=E1+Delta;
                    accept_all=[accept_all 1];
                else
                    if (rand< exp(-delta))     %accept x_new with probability exp -delta
                        x=x_new;
                        E1=E1+Delta;
                        accept_all=[accept_all 1];
                    else
                        accept_all=[accept_all 0];
                    end
                end;
			case 2,
				% choose new x by flipping random bits i,j
				% perform Metropolis Hasting step
                i = randi([1 n],1,1);
                j = randi([1 n],1,1);
                x_new = x;
                x_new(i) = -1 * x_new(i);
                if (j~=i)
                    x_new(j) = -1 * x_new(j);
                    a = x_new(i)-x(i);
                    b = x_new(j)-x(j);
                    Delta = - (a * x * w(:,i)+b * x * w(:,j)) - 0.5 * (a*b*w(j,i)+b*a*w(i,j));
                else
                    Delta = - (x_new(i) - x(i)) * x * w(:,i);
                end;
                delta=beta*Delta;
                if (delta < 0)
                    x=x_new;                   %accept x_new
                    E1=E1+Delta;
                    accept_all=[accept_all 1];
                else
                    if (rand< exp(-delta))     %accept x_new with probability exp -delta
                        x=x_new;
                        E1=E1+Delta;
                        accept_all=[accept_all 1];
                    else
                        accept_all=[accept_all 0];
                    end
                end;
			end;
			% E1 is energy of new state
			E_all(t1)=E1;
		end;
		E_outer(t2)=mean(E_all);
		E_bar(t2)=std(E_all);
		[t2 beta E_outer(t2) E_bar(t2)] % observe convergence
	end;
	E_min=E_all(1) % minimal energy

% 	plot(Beta_all,E_outer(1:t2),Beta_all,E_bar(1:t2))
% 	plot(1:t2,E_outer(1:t2),1:t2,E_bar(1:t2))
	plot(HeatCapacity_all)
	xlabel('Temp')
	ylabel('Energy')
end;


