
clear,close;

Original_image_dir = '/media/nankaingy/zalick/PGSmoothing/RTV_stru';
fpath   = fullfile(Original_image_dir, '*.jpg');
im_dir  = dir(fpath);
im_num     = length(im_dir);

method = 'RGF';
for i = 1:im_num
    I = im2double(imread(fullfile(Original_image_dir, im_dir(i).name)));
    S = regexp(im_dir(i).name, '\.', 'split');
    sI = RollingGuidanceFilter(I,3,0.05,4);
    fprintf('%s is done!\n', im_dir(i).name);
    outname = sprintf(['/media/nankaingy/zalick/PGSmoothing/stru_Methods/RGF/' S{1} '_' method '.jpg']);
    imwrite(sI, outname);
end