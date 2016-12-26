rand('state',0)
n=50;
grid_x=10;
% grid=0; %1D grid
% grid=1; %2D grid quadratic
grid=2; %2D grid triangular


p=n;
rand('state',0);
w=sprandsym(n,p);
w=(w>0)-(w<0); % this choice defines a frustrated system
% w=(w>0); % this choice defines a ferro-magnetic (easy) system
% w=zeros(n,p);
for i=1:n
    row=w(i,:);
    neigh=Neigh(n,i,grid,grid_x)
    row(neigh)=0;
    w(i,:)=w(i,:)-row;
%     w(i,neigh)=1;
end;


w=w-diag(diag(w));

% setting the weights of spins out of the neighbourhood to 0


