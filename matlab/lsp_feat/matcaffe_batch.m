% function feat = matcaffe_batch(list_im, use_gpu)
function matcaffe_batch(list_im, use_gpu)

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
  imdir = '/home/wyang/github/caffe/matlab/testdata/';
  imlist = dir([imdir '*.jpg']);
  for i = 1:length(imlist)
      [p, imname, ext] = fileparts(imlist(i).name); imname
      list_im{i} = [imdir imname ext];
  end
  [imh, imw, ~] = size(imread(list_im{4}));
end
if ischar(list_im)
    %Assume it is a file contaning the list of images
    filename = list_im;
    list_im = read_cell(filename);
end
% Adjust the batch size and dim to match with models/bvlc_reference_caffenet/deploy.prototxt
batch_size = 10;
dim = 5;
nchannel = 32;
disp(list_im)
if mod(length(list_im),batch_size)
    warning(['Assuming batches of ' num2str(batch_size) ' images rest will be filled with zeros'])
end

% init caffe network (spews logging info)
if exist('use_gpu', 'var')
  matcaffe_init(use_gpu);
else
  matcaffe_init(1, '../../examples/lsp_patch_const/lsp-xianjie-deploy2.prototxt','/home/wyang/Data/cache/caffe/LSP_P26_K17_patch/models/model-01-16/lsp-patch-train_iter_300000.caffemodel');
end


% prepare input

num_images = length(list_im);
feat = zeros(dim, dim, nchannel, ceil(length(list_im)/batch_size)*batch_size,'single');
labels = zeros(1, ceil(length(list_im)/batch_size)*batch_size);

num_batches = ceil(length(list_im)/batch_size)
initic=tic;
for bb = 1 : num_batches
    batchtic = tic;
    range = 1+batch_size*(bb-1):min(num_images,batch_size * bb);
    featidx =  1+batch_size*(bb-1):batch_size * bb;
    tic
%     labels(:, range) = prepare_batch_label(warped(range),batch_size);
    input_data = prepare_batch(list_im(range),batch_size, imw, imh);
    size(input_data)
    toc, tic
    fprintf('Batch %d out of %d %.2f%% Complete ETA %.2f seconds\n',...
        bb,num_batches,bb/num_batches*100,toc(initic)/bb*(num_batches-bb));
    output_data = caffe('forward', {input_data});
    size(output_data{1})
    toc
%     feat(:, :, :, featidx) = squeeze(output_data{1});
    toc(batchtic)
end
toc(initic);