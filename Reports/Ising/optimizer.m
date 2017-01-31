% problem definition
% minimize E= 0.5 x^T w x, with x=x_1,...,x_n and x_i=0,1
% w is a symmetric real n x n matrix with zero diagonal


if(exist('w','var') ~=1 || exist('n','var')~=1 || exist('grid_x','var')~=1)
    error('Please run makedata.m with the neigbourhood function of your choice.')
end;

% METHOD='iter';
% NEIGHBORHOODSIZE=1;
n_restart =500;
colormap('Gray');
gif_fps = 24;
% Define string variable that holds the filename of ising gif
video_filename = 'ising.gif';
fh = figure(1);
create_gif=1; % show ising state visualy or not


tx = zeros(1,n_restart);
switch METHOD,
case 'iter'
    disp('Iterative Method')
    tic
	E_min = 1000;
	E_min_all = [];
	for t=1:n_restart,

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
                    end;
                end;
                % Terminate only if no new minimums are not found in the neighbourhood.
                if (i==n)
                    flag = 0;
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
                            Delta = - (a*A(:,i)+b*A(:,j)) - 0.5 * (a*b*w(j,i)+b*a*w(i,j));
                        else
                            Delta = - (xN(i) - x(i)) * A(i);
                        end;
                        if (Delta<0)
                            E1 = E1 + Delta;
                            x=xN;
                            innerFlag=0;
                            E_all = [E_all E1];
                        end;
                    end;
%                     if (innerFlag==0) break; end;
                end;
                if (i==n && j==n) flag = 0; end;
			end;
		end;
		if (E_min == E1)
            E_min = min(E_min,E1);
            E_min_all = [E_min_all E_min];
		    break;
		end;
		if(create_gif==1)
            imagesc(reshape(x,[grid_x,n/grid_x])');
            xlabel(sprintf( 'E = %0.2f',   E1/n));
            drawnow;
            frame = getframe(fh);
            % Turn screenshot into image
            im = frame2im(frame);
            % Turn image into indexed image (the gif format needs this)
            [imind,cm] = rgb2ind(im,256);
            % If first loop iteration: Create the file, else append to it
            if t==1
                imwrite(imind,cm,video_filename,'gif', 'Loopcount',inf);
            else
                imwrite(imind,cm,video_filename,'gif','WriteMode','append','DelayTime',1/gif_fps);
            end
        end;
		E_min = min(E_min,E1);
		E_min_all = [E_min_all E_min];
	end;
	E_min
	t
	tx = toc;
    plot(1:length(E_all),E_all)
	xlabel('Iteration')
	ylabel('Min Energy')
case 'sa'
	% initialize
    disp('Simulated Anealing')
	x = 2*(rand(1,n)>0.5)-1;
	E1 = E(x,w);
	E_outer=zeros(1,100);	%stores mean energy at each temperature
	E_bar=zeros(1,100);		% stores std energy at each temperature


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
	T1=2000; % length markov chain at fixed temperature
	factor=1.05 ; % increment of beta at each new chain

	beta=beta_init;
	E_bar(1)=1;
	t2=1;
	Beta_all=[beta];
	HeatCapacity_all=[];
	accept_all=[];
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
		if(create_gif==1)
            imagesc(reshape(x,[grid_x,n/grid_x])');
            xlabel(sprintf('T = %0.2f, E = %0.2f', 1/beta,  E1/n));
            drawnow;
            frame = getframe(fh);
            % Turn screenshot into image
            im = frame2im(frame);
            % Turn image into indexed image (the gif format needs this)
            [imind,cm] = rgb2ind(im,256);
            % If first loop iteration: Create the file, else append to it
            if t2==2
                imwrite(imind,cm,video_filename,'gif', 'Loopcount',inf);
            else
                imwrite(imind,cm,video_filename,'gif','WriteMode','append','DelayTime',1/gif_fps);
            end
        end;
		E_outer(t2)=mean(E_all);
		E_bar(t2)=std(E_all);
		[t2 beta E_outer(t2) E_bar(t2)] % observe convergence
	end;
	E_min=E_all(1) % minimal energy

end;


