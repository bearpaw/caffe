function weight_surgery(im, use_gpu, protopath, modelpath, rmean, gmean, bmean, sbin)
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
addpath('../tools/');
if nargin < 1
  % For test purposes
  impath = '/home/wyang/Datasets/lsp_dataset/images/im1001.jpg';
  im = imread(impath);
end
if nargin < 2
    use_gpu = 1;
end
if nargin < 3
    protopath = '/home/wyang/github/caffe/examples/lsp_window_data/conv5-96c/lsp-xianjie-train-test-deploy-conv.prototxt';
end
if nargin < 4
    modelpath = '/home/wyang/Data/cache/caffe/LSP_P26_K17_patch/models/train-12-Feb-2015-conv5-96-2/lsp-patch-train_iter_55000.caffemodel';
end
if nargin < 5
    rmean = 115;
end
if nargin < 6
    gmean = 108;
end
if nargin < 7
    bmean = 100;
end
if nargin < 6
    sbin = 4;
end


tic;
% prepare input
im = prepare_image(int16(im), rmean, gmean, bmean, sbin);
% create prototxt
protopath = prepare_proto(im, protopath);
% init caffe network (spews logging info)
matcaffe_init(use_gpu, protopath, modelpath, 1);
% extract output
feat = caffe('forward', {im}); feat = feat{1};
feat = double(permute(feat,[2 1 3]) );
toc;
params = caffe('get_weights');
for i = 1:length(params)
    weights = params(i).weights{1};
    bias = params(i).weights{2};
    layername = params(i).layer_names;  
    [k, s, indim, outdim] = size(weights);
    initweightstd = sqrt(2/(k*s*indim*outdim));
    initweightstdfwd = sqrt(2/(k*s*indim));
    initweightstdbwd = sqrt(2/(k*s*outdim));
    fprintf('%s weights: min: %.4f | max %.4f | init std fwd: %.4f,  | init std bwd: %.4f | init std all: %.4f\n', layername, min(weights(:)), max(weights(:)), initweightstdfwd, initweightstdbwd, initweightstd);
    hist(weights(:), 32); pause; close;
    f = display_network_4D(weights, 'conv');
    title(layername);
    pause; close;
end
end

% ------------------------------------------------------------------------
function im = prepare_image(im, rmean, gmean, bmean, sbin)
% ------------------------------------------------------------------------

% resize image
[dimy, dimx, ~] = size(im);
blocky = round(dimy / sbin);
blockx = round(dimx / sbin);
visibley = blocky*sbin;
visiblex = blockx*sbin;    
im = double(imresize(im, [visibley, visiblex]));
    
% minus image mean
im(:, :, 1) = im(:, :, 1)- rmean;
im(:, :, 2) = im(:, :, 2)- gmean;
im(:, :, 3) = im(:, :, 3)- bmean;
im = single(im);

% imshow(uint8(im)); pause;
% Transform GRAY to RGB
if size(im,3) == 1
    im = cat(3,im,im,im);
end
end

% ------------------------------------------------------------------------
function protopath = prepare_proto(im, prototxt)
% ------------------------------------------------------------------------
[h, w, ~] = size(im);
curDir   = pwd;
tmppath = 'tmp.txt';
protopath = [curDir '/deploy.prototxt'];
f = fopen(tmppath, 'w');
fprintf(f, 'name: "XianjieNet"\n');
fprintf(f, 'input: "data"\n');
fprintf(f, 'input_dim: 1\n');
fprintf(f, 'input_dim: 3\n');
fprintf(f, 'input_dim: %d\n', h);
fprintf(f, 'input_dim: %d\n', w);
fclose(f);
[status,cmdout] = unix(sprintf('cat %s %s > %s', tmppath, prototxt, protopath));
end