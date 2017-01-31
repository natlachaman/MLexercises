rand('state',0)
% n=25;
% grid_x=5; % X dimension of lattice, it should be a divisor of n
% grid=0; %1D grid
% grid=1; %2D grid quadratic
% grid=2; %2D grid triangular
% grid=3; %no neighbourhood

% J=1; %set to 1 or -1 if you dont want random J
% frustr=1;



p=n;


if(frustr==0)
    w=zeros(n,p); % use this instead of the above random generation to avoid memory problems if J isnt random.
else
    w=sprandsym(n,p);
    w=(w>0)-(w<0);
    J=0;
end;
% Creating the Neighbourhoods
if(grid~=3)
    for i=1:n
        row=w(i,:);
        neigh=Neigh(n,i,grid,grid_x);
        row(neigh)=0;
        w(i,:)=w(i,:)-row;
        if(J~=0)
            w(i,neigh)=J;
        end;
    end;
end;

w=w-diag(diag(w));




