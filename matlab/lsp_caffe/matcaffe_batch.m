function [scores,list_im] = matcaffe_batch(list_im, use_gpu)
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
  imdir = '/home/wyang/Data/Code/pose/PE1.41-milestone1/';
  load(['/home/wyang/Data/Code/pose/PE1.41-milestone1/cache/LSP_P26_K7_train_warped.mat'], 'warped');
  shuffle_map = randperm(length(warped));
  warped(shuffle_map) = warped;
  warped = warped(1: 100);
  list_im = cell(length(warped), 1);
  for i = 1:length(warped)
      list_im{i} = [imdir warped(i).im];
  end
end
if ischar(list_im)
    %Assume it is a file contaning the list of images
    filename = list_im;
    list_im = read_cell(filename);
end
% Adjust the batch size and dim to match with models/bvlc_reference_caffenet/deploy.prototxt
batch_size = 10;
dim = 183;
disp(list_im)
if mod(length(list_im),batch_size)
    warning(['Assuming batches of ' num2str(batch_size) ' images rest will be filled with zeros'])
end

% init caffe network (spews logging info)
if exist('use_gpu', 'var')
  matcaffe_init(use_gpu);
else
  matcaffe_init(1, '../../examples/lsp_patch_const/lsp-xianjie-deploy.prototxt','/home/wyang/Data/cache/caffe/LSP_P26_K17_patch/models/model-01-16/lsp-patch-train_iter_1000.caffemodel');
end


% prepare input

num_images = length(list_im);
scores = zeros(dim,num_images,'single');
labels = zeros(1, num_images);

num_batches = ceil(length(list_im)/batch_size)
initic=tic;
for bb = 1 : num_batches
    batchtic = tic;
    range = 1+batch_size*(bb-1):min(num_images,batch_size * bb);
    tic
    labels(:, range) = prepare_batch_label(warped(range),batch_size);
    input_data = prepare_batch(list_im(range),batch_size);
    toc, tic
    fprintf('Batch %d out of %d %.2f%% Complete ETA %.2f seconds\n',...
        bb,num_batches,bb/num_batches*100,toc(initic)/bb*(num_batches-bb));
    output_data = caffe('forward', {input_data});
    toc
    output_data = squeeze(output_data{1});
    scores(:,range) = output_data(:,mod(range-1,batch_size)+1);
    toc(batchtic)
end
toc(initic);
% Top K Accuracy
top_k = 5;
[s, idx] = sort(scores);
accuracy = 0;

top_k_idx = idx(1:top_k, :);
for i = 1:num_images
    if ~isempty(find(labels(i) == top_k_idx(:, i)))
        accuracy = accuracy + 1;
    end
end
accuracy = accuracy / num_images;

% if exist('filename', 'var')
%     save([filename '.probs.mat'],'list_im','scores','-v7.3');
% end



