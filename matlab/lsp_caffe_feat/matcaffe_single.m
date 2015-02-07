function feat = matcaffe_single(im, protofile, caffemodel, use_gpu, batch_size, dim, nchannel, imagemean)
sbin = 4;
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
if nargin < 1
   im = imresize(imread('peppers.png'), [28, 28]);
end
if nargin < 2
    protofile = '../../examples/lsp_patch_const/lsp-xianjie-deploy.prototxt';
end
if nargin < 3
    caffemodel = '/home/wyang/Data/cache/caffe/LSP_P26_K17_patch/models/model-01-16/lsp-patch-train_iter_300000.caffemodel';
end
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
    imagemean = single(zeros(size(im)));
end

% init caffe network (spews logging info)
matcaffe_init(use_gpu, protofile, caffemodel);

% prepare input
feat = zeros(dim, dim, nchannel, 1,'single');
input_data = prepare_image(im, imagemean);

tic;
tmp = caffe('forward', input_data);
toc;



% ------------------------------------------------------------------------
function images = prepare_image(im, imagemean)
% ------------------------------------------------------------------------
images = zeros(size(im, 1), size(im, 2), 3, 2,'single');
try
    % resize image
    [dimy, dimx, ~] = size(im);
    blocky = round(dimy / sbin);
    blockx = round(dimx / sbin);
    visibley = blocky*sbin;
    visiblex = blockx*sbin;    
    im = double(imresize(im, [visibley, visiblex]));
    
    % prepare caffe input
    im = single(im);
    im = im - imagemean;
    % Transform GRAY to RGB
    if size(im,3) == 1
        im = cat(3,im,im,im);
    end
    % permute from RGB to BGR
    im = im(:,:,[3 2 1]);
    images(:,:,:,1) = im;
catch
    warning('Problems with file',image_files{i});
end