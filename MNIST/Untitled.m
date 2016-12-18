a=cputime;


i = 0;
for t = 0:.01:10000
    i = i + 1;
    y(i) = sin(t);
end


b=cputime;
disp(b-a)

clear all


a=cputime;

t = 0:.01:10000;
y = sin(t);
b=cputime;
disp(b-a)