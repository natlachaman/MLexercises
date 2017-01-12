files = dir('*.mat');


for i =  1:length(files)-3
    fname =  files(i).name;
    load(fname,'err')
end





weights = random(neurons,neurons);
weights = gpuArray(weights);