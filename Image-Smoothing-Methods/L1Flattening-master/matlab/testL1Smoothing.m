clear,close;
addpath('./../src');
Original_image_dir = '/media/nankaingy/zalick/PGSmoothing/My_dataset/Benchmark_Final/data';
fpath   = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num     = length(im_dir);

method = 'L1';

% use default parameters for image flattening
param = struct(); 
param.local_param.edge_preserving = true; 

for i = 1:im_num
    I = imread(fullfile(Original_image_dir, im_dir(i).name));
    S = regexp(im_dir(i).name, '\.', 'split');
    sI = l1flattening(I, param);
    fprintf('%s  %d is done!\n', im_dir(i).name, i);
    outname = sprintf(['/media/nankaingy/zalick/PGSmoothing/My_dataset/Benchmark_Final/' method '/' S{1} '_' method '.png']);
    imwrite(sI, outname);
end

