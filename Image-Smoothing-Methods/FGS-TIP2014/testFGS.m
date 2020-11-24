clear,close;

Original_image_dir = '/media/nankaingy/zalick/PGSmoothing/My_dataset/Benchmark_Final/data';
fpath   = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num     = length(im_dir);

method = 'FGS';
for i = 1:im_num
    I = imread(fullfile(Original_image_dir, im_dir(i).name));
    S = regexp(im_dir(i).name, '\.', 'split');    
    F = FGS(I, 0.1, 30^2, [], 3, 4);
    fprintf('%s is done!\n', im_dir(i).name);
    outname = sprintf(['/media/nankaingy/zalick/PGSmoothing/My_dataset/Benchmark_Final/FGS/' S{1} '_' method '.png']);
    imwrite(F, outname);
end