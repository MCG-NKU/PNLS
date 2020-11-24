clear,close;

Original_image_dir = '/media/nankaingy/zalick/PGSmoothing/Picked_Image/RTV_stru/TMM';
fpath   = fullfile(Original_image_dir, '*.jpg');
im_dir  = dir(fpath);
im_num     = length(im_dir);

method = 'L0';
for i = 1:im_num
    I = imread(fullfile(Original_image_dir, im_dir(i).name));
    S = regexp(im_dir(i).name, '\.', 'split');
%     sI = L0Smoothing(I,0.01);
    sI = L0Smoothing(I,0.02);
    fprintf('%s is done!\n', im_dir(i).name);
    outname = sprintf(['/media/nankaingy/zalick/PGSmoothing/Picked_Image/RTV_stru/L0' S{1} '_' method '.jpg']);
    imwrite(sI, outname);
end