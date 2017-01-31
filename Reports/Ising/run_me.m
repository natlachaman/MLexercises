METHOD='sa';
NEIGHBORHOODSIZE=1;
grid=2; %grid type 2=triangular 3 for no for grid-wide neigbourhood
J=1; %J=1 only if frustr=0
grid_x=10; %grid dimension
n=200; %spin number
frustr=0; %frustration setting
makedata;
optimizer;



% 	Plot Perspin Energy or StandardDeviation of Energy

errorbar(Beta_all.^-1,E_outer(1:t2)/n,E_bar(1:t2)/n,'-.k.')
%     plot(Beta_all.^-1,E_bar(1:t2)/n)
set(gca,'xscale','log')
xlim([0,20]);
xlabel('Temperature');
% 	ylabel('sd of Energy');
ylabel('Energy per spin');
if (grid==2)
    title_grid='Triangular';
elseif(grid==1)
    title_grid='Rectangular';
else
    title_grid='1D';
end;
if (J==0)
    title_J='J=-1/1';
else
    title_J=strcat('J=',int2str(J));
end;
title(sprintf('%s %dx%d Lattice Ising %s',title_grid,grid_x,n/grid_x,title_J ));






