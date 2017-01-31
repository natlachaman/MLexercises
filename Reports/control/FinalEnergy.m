function energy = FinalEnergy(x_0,v_0,lambda)

d_t=0.05;
nu=0.1;
T=10;
R=1;
g=5;
v=v_0;
x=x_0;
X_ALL=[];
u=0;
for t=0:d_t:T-d_t
    dl=dL(x);
    F=-g*dl/sqrt(1+dl^2);
    dv=F*d_t+u*d_t+normrnd(0,sqrt(nu*d_t));
    v=v+dv;
    dx=v*d_t;
    x=x+dx;
    X_ALL=[X_ALL x];
end;
if max(X_ALL)>2 || min(X_ALL)<-2
    energy=1/lambda;
else
    energy=0/lambda;
end;