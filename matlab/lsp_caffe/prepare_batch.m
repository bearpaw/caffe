% ------------------------------------------------------------------------
function images = prepare_batch(image_files,batch_size)
% ------------------------------------------------------------------------
num_images = length(image_files);
if nargin < 2
    batch_size = num_images;
end

IMAGE_DIM = 28;

num_images = length(image_files);
images = zeros(IMAGE_DIM,IMAGE_DIM,3,batch_size,'single');

for i=1:num_images
    % read file
    fprintf('%c Preparing %s\n',13,image_files{i});
    try
        im = imread(image_files{i});
        % resize to fixed input size
        im = single(im);
        im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
        % Transform GRAY to RGB
        if size(im,3) == 1
            im = cat(3,im,im,im);
        end
        % permute from RGB to BGR
        im = im(:,:,[3 2 1]);
        images(:,:,:,i) = im;
    catch
        warning('Problems with file',image_files{i});
    end
end