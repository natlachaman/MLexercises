
x_0=0.5;
v_0=0;
d_t=0.01;
nu=0.1;
T=10;
R=1;
g=9;

v=v_0;
x=x_0;
X_ALL=[];
u=0;%no optimal control
for t=0:d_t:T-d_t
    dl=dL(x);
    F=-g*dl/sqrt(1+dl^2);
    dv=F*d_t+u*d_t+normrnd(0,sqrt(nu*d_t)); %calculating dv with normal noise
    v=v+dv;
    dx=v*d_t;
    x=x+dx;
    X_ALL=[X_ALL x];% calculate and append the new x in time t
    if max(X_ALL)>2 || min(X_ALL)<-2
        if max(X_ALL)>2
            mm=max(X_ALL)
        else
            mm=min(X_ALL)
        end
        disp (sprintf('Made it out at x=%0.4f!! v_0=%0.1f t=%d x_0=%0.2f \\nu=%0.3f',mm,v_0,t,x_0,nu))
        break;
    end;
end;

xx=-2:0.01:2;
plot(xx,L(xx),X_ALL,L(X_ALL),'r');
axis([-2 2 -2 -.9])
title(sprintf('v_0=%0.1f T=%d x_0=%0.2f',v_0,T,x_0));