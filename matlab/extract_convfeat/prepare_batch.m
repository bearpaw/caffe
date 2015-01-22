% ------------------------------------------------------------------------
function images = prepare_batch(image_files,batch_size, imagemean)
% ------------------------------------------------------------------------
num_images = length(image_files);
if nargin < 2
    batch_size = num_images;
end

IMAGE_DIM = size(image_files, 1);

num_images = size(image_files, 4);
images = zeros(IMAGE_DIM,IMAGE_DIM,3,batch_size,'single');

for i=1:num_images
    % read file
%     fprintf('%c Preparing %s\n',13,image_files{i});
    try
        im = single(image_files(:, :, :, i));
        im = im - imagemean;
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