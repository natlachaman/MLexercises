

d_t=0.01
plot_=0;
for x_0=[0]
    for T=[1,10,50,100]
        for nu=[0.01,0.1,0.5]
            plot_=plot_+1;
            time_discretation=0:d_t:T-d_t;
            X_ALL=[];
            DU_ALL=[];
            x=x_0;
            u=0;
            for t=time_discretation %move through time and calculate the steps
                u_star=(tanh(x/(nu*(T-t)))-x)/(T-t);
                d_x=u_star*d_t+normrnd(0,sqrt(nu*d_t));
                DU_ALL=[DU_ALL u_star];
                x=x+ d_x;
                X_ALL=[X_ALL x];
            end;
            ax=subplot(4,3,plot_);
            plot(time_discretation,X_ALL,'k',time_discretation,DU_ALL,'r');
            hline=refline(ax,[0 1]);
            hline=refline(ax,[0 -1]);
            title(sprintf('x0=%0.1f T=%d \\nu=%0.2f',x_0,T,nu));
        end;
    end;
end;

