% Demo script
% Uncomment each case to see the results 

clear,close;

Original_image_dir = '/media/nankaingy/zalick/PGSmoothing/RTV_stru';
fpath   = fullfile(Original_image_dir, '*.jpg');
im_dir  = dir(fpath);
im_num     = length(im_dir);

method = 'RTV';
for i = 1:im_num
    I = imread(fullfile(Original_image_dir, im_dir(i).name));
    S = regexp(im_dir(i).name, '\.', 'split');
    sI = tsmooth(I,0.015,3);
    fprintf('%s is done!\n', im_dir(i).name);
    outname = sprintf(['/media/nankaingy/zalick/PGSmoothing/stru_Methods/RTV/' S{1} '_' method '.jpg']);
    imwrite(sI, outname);
end

% I = (imread('imgs/crossstitch.jpg'));
% S = tsmooth(I,0.015,3);
% figure, imshow(I), figure, imshow(S);

% I = (imread('imgs/graffiti.jpg'));
% S = tsmooth(I,0.015,3);
% figure, imshow(I), figure, imshow(S);

% I = (imread('imgs/crossstitch.jpg'));
% S = tsmooth(I,0.015,3);
% figure, imshow(I), figure, imshow(S);

% I = (imread('imgs/mosaicfloor.jpg'));
% S = tsmooth(I, 0.01, 3, 0.02, 5); 
% figure, imshow(I), figure, imshow(S);






