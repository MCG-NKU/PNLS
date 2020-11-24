clear,close;

Original_image_dir = '/media/nankaingy/zalick/PGSmoothing/My_dataset/My_Benchmark/data';
fpath   = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num     = length(im_dir);

method = 'BF';
for i = 1:im_num
    I = imread(fullfile(Original_image_dir, im_dir(i).name));
    S = regexp(im_dir(i).name, '\.', 'split');
    sI = bilateralFilter(I);
    %sI = imbilatfilt(I);
    fprintf('%s is done!\n', im_dir(i).name);
    outname = sprintf(['/media/nankaingy/zalick/PGSmoothing/My_dataset/My_Benchmark/BF/' S{1} '_' method '.png']);
    imwrite(sI, outname);
end