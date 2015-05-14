function feat = matcaffe_batch(images, protofile, caffemodel, use_gpu, batch_size, dim, nchannel, imagemean)
% scores = matcaffe_batch(list_im, use_gpu)
%
% Demo of the matlab wrapper using the ILSVRC network.
%
% input
%   list_im  list of images files
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   scores   1000 x num_images ILSVRC output vector
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  scores = matcaffe_batch({'peppers.png','onion.png'});
%  scores = matcaffe_batch('list_images.txt', 1);
if nargin < 4
    use_gpu = 0;
end
if nargin < 5
    batch_size = 10;
end
if nargin < 6
    dim = 5;
end
if nargin < 7
    nchannel = 32;
end
if nargin < 8
    imagemean = zeros(size(images(:, :, :, 1)));
end

% init caffe network (spews logging info)
matcaffe_init(use_gpu, protofile, caffemodel, 1);

% prepare input
num_images = size(images, 4);
feat = zeros(dim, dim, nchannel, ceil(num_images/batch_size)*batch_size,'single');
num_batches = ceil(num_images/batch_size);

initic=tic;
for bb = 1 : num_batches
    range = 1+batch_size*(bb-1):min(num_images,batch_size * bb);
    featidx =  1+batch_size*(bb-1):batch_size * bb;

    % prepare images
    minibatch = prepare_batch(images(:, :, :, range),batch_size, imagemean);
    output_data = caffe('forward', {minibatch});
    feat(:, :, :, featidx) = squeeze(output_data{1});
end
feat = double(feat(:, :, :, 1:num_images));
feat = double(permute(feat,[2 1 3 4]) );
toc(initic);

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
    try
        im = int16(image_files(:, :, :, i)) - imagemean;
        im = single(im);
        % Transform GRAY to RGB
        if size(im,3) == 1
            im = cat(3,im,im,im);
        end
        images(:,:,:,i) = im;
    catch
        warning('Problems with file',image_files{i});
    end
end