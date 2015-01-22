function feat = matcaffe_single(im, use_gpu, protopath, modelpath)
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
  % For test purposes
  impath = '/home/wyang/github/caffe/matlab/testdata/pyra_10.jpg';
  im = imread(impath);
end
if nargin < 2
    use_gpu = 1;
end
if nargin < 3
    protopath = 'lsp-xianjie-deploy.prototxt';
end
if nargin < 4
    modelpath = '/home/wyang/Data/cache/caffe/LSP_P26_K17_patch/models/model-01-16/lsp-patch-train_iter_300000.caffemodel';
end


tic;
% prepare input
input_data = prepare_image(im);
% create prototxt
protopath = prepare_proto(im, protopath);
% init caffe network (spews logging info)
matcaffe_init(use_gpu, protopath, modelpath, 1);
% extract output
feat = caffe('forward', {input_data});
toc;
size(feat{1})

end

% ------------------------------------------------------------------------
function im = prepare_image(im)
% ------------------------------------------------------------------------
[h, w, ~] = size(im)
images = zeros(h,w,3,2,'single');
im = single(im);
% Transform GRAY to RGB
if size(im,3) == 1
    im = cat(3,im,im,im);
end
% permute from RGB to BGR
im = im(:,:,[3 2 1]);
images(:,:,:,1) = im;
images(:,:,:,2) = im;
end

% ------------------------------------------------------------------------
function protopath = prepare_proto(im, prototxt)
% ------------------------------------------------------------------------
[h, w, ~] = size(im);
tmppath = 'tmp.txt';
protopath = 'deploy.prototxt';
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