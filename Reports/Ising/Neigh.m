function Neighbours=Neigh(n,i,grid,dimx)
LR=[];
TB=[];
dimy=n/dimx;
switch grid,
case 0,
%     One dimensional Grid
    if (i==1)
        LR=[LR n];
        LR=[LR i+1];
    elseif (i==n)
        LR=[LR 1];
        LR=[LR i-1];
    else
        LR=[LR i+1];
        LR=[LR i-1];
    end;
case 1,
%     Square Grid
    borderx=0;
    bordery=0;
    if(rem(i,dimx)==0)
        LR=[LR i-1];%left
        LR=[LR i-dimx+1];%right
        borderx=1;
    elseif(rem(i,dimx)==1)
        LR=[LR i+dimx-1];%left
        LR=[LR i+1];%right
        borderx=1;
    end
    if(i<=dimx)
        TB=[TB n-dimx+i];%top
        TB=[TB i+dimx];%bottom
        bordery=1;
    elseif(i>n-dimx)
        TB=[TB i-dimx];%top
        TB=[TB i-n+dimx];%bottom
        bordery=1;
    end
    if(borderx==0)
        LR=[LR i-1];%left
        LR=[LR i+1];%right
    end;
    if(bordery==0)
        TB=[TB i-dimx];%top
        TB=[TB i+dimx];%bottom
    end;
case 2,
%     Triangular Grid
    if(rem(i,2)==0)
        LR=Neigh(n,i,1,dimx);
        TB=Neigh(n,LR(4),1,dimx);
        TB=TB([1,2]);
    else
        LR=Neigh(n,i,1,dimx);
        TB=Neigh(n,LR(3),1,dimx);
        TB=TB([1,2]);
    end;
end;
Neighbours=[LR TB];