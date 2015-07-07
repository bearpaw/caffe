function weight_surgery(impath, use_gpu, protopath, modelpath, rmean, gmean, bmean, sbin)
% Demo of visualization of caffe model
%
% input
%   im  - test image
%   use_gpu  - 1 to use the GPU, 0 to use the CPU
%   protopath - deploy file WITHOUT HEAD (INPUT_DIM)
%   modelpath  - model path
%   rmean  - mean red
%   gmean  - mean green
%   bmean  - mean blue
%   sbin  - number of bins of weights
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  scores = matcaffe_batch({'peppers.png','onion.png'});
%  scores = matcaffe_batch('list_images.txt', 1);
addpath('../caffe/');

load('/home/wyang/Data/Code/pose/chen_nips14_full/cache/lsp/regression_test_patches.mat'); % impatches

if nargin < 1
  % For test purposes
  impath = '/home/wyang/Data/Code/pose/chen_nips14_full/regression_test.jpg';
end
if nargin < 2
    use_gpu = 1;
end
if nargin < 3
    protopath = '/home/wyang/Data/Code/pose/chen_nips14_full/external/my_models/lsp_regression_4096/lsp_deploy.prototxt';
end
if nargin < 4
    modelpath = '/home/wyang/Data/Code/pose/chen_nips14_full/external/my_models/lsp_regression_4096/cache/lsp_iter_50000.caffemodel';
end
if nargin < 5
    rmean = 128;
end
if nargin < 6
    gmean = 128;
end
if nargin < 7
    bmean = 128;
end
if nargin < 6
    sbin = 4;
end


tic;

% init caffe network (spews logging info)
caffe_init(use_gpu, protopath, modelpath, 1);
% extract output

delta = {};
for i = 1:length(impatches)
  i
  impyra = impatches{i};
  resp = caffe('forward', {impyra}); resp = squeeze( resp{1} );
  delta{i} = resp;
%   params = caffe('get_weights');
%   for i = 1:length(params)
%       weights = params(i).weights{1};
%       bias = params(i).weights{2};
%       layername = params(i).layer_names;  
%       [k, s, indim, outdim] = size(weights);
%       initweightstd = sqrt(2/(k*s*indim*outdim));
%       initweightstdfwd = sqrt(2/(k*s*indim));
%       initweightstdbwd = sqrt(2/(k*s*outdim));
%       fprintf('%s weights: min: %.4f | max %.4f | init std fwd: %.4f,  | init std bwd: %.4f | init std all: %.4f\n', layername, min(weights(:)), max(weights(:)), initweightstdfwd, initweightstdbwd, initweightstd);
%       hist(weights(:), 32); pause; close;
%       f = display_network_4D(weights, 'conv');
%       title(layername);
%       pause; close;
%   end
end
save('delta.mat', 'delta');


toc;
params = caffe('get_weights');
% for i = 1:length(params)
%     weights = params(i).weights{1};
%     bias = params(i).weights{2};
%     layername = params(i).layer_names;  
%     [k, s, indim, outdim] = size(weights);
%     initweightstd = sqrt(2/(k*s*indim*outdim));
%     initweightstdfwd = sqrt(2/(k*s*indim));
%     initweightstdbwd = sqrt(2/(k*s*outdim));
%     fprintf('%s weights: min: %.4f | max %.4f | init std fwd: %.4f,  | init std bwd: %.4f | init std all: %.4f\n', layername, min(weights(:)), max(weights(:)), initweightstdfwd, initweightstdbwd, initweightstd);
%     hist(weights(:), 32); pause; close;
%     f = display_network_4D(weights, 'conv');
%     title(layername);
%     pause; close;
% end
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
protopath = [curDir '/tmp_deploy.prototxt'];
f = fopen(tmppath, 'w');
fprintf(f, 'name: "XianjieNet"\n');
fprintf(f, 'input: "data"\n');
fprintf(f, 'input_dim: 1\n');
fprintf(f, 'input_dim: 3\n');
fprintf(f, 'input_dim: %d\n', h);
fprintf(f, 'input_dim: %d\n', w);
fclose(f);
[status,cmdout] = unix(sprintf('cat %s %s > %s', tmppath, prototxt, protopath));
unix(['rm -rf ' tmppath]);
end