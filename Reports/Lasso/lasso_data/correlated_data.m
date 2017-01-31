% example 1 from Zhao and Yu 2006
    n=3;
    p=1000;     % train data size
    w=[2,3,0];  % example 1a
    %w=[-2,3,0];    % example 1b
    sigma=1;
    x(1:2,:)=randn(2,p);
    x(3,:)=2/3*x(1,:)+2/3*x(2,:)+1/3*randn(1,p);
    y=w*x+randn(1,p);
