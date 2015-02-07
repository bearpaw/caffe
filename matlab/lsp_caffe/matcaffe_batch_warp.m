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
    try
        load('warp_data.mat', 'data', 'labels');
    catch
        % For test purposes
        datadir = '/home/wyang/Data/Code/pose/PE1.41-milestone1/';
        load('/home/wyang/Data/Code/pose/PE1.41-milestone1/cache/LSP_K7/wraped-images.mat', 'testwrap');
        load /home/wyang/Data/Code/pose/PE1.41-milestone1/cache/LSP_K7/lsp-channel-const-mean.mat;
    %     testwrap = testwrap(1:1000);
        data = zeros(28, 28, 3, length(testwrap));
        labels = zeros(1, length(testwrap));
        stime = tic;
        for i = 1:length(testwrap)
            % !!!!! Read as int16 and minus mean
            data(:, :, :, i) = int16(imread([datadir testwrap(i).im])) - meanimg;
            if testwrap(i).part == 0 && testwrap(i).cluster == 0
                labels(i) = 0;
            else
                labels(i) = (testwrap(i).part-1)*7 + testwrap(i).cluster;
            end
        end
        toc(stime)
        save('warp_data.mat', 'data', 'labels');
    end
end
% Adjust the batch size and dim to match with models/bvlc_reference_caffenet/deploy.prototxt
batch_size = 100;
dim = 183;

prototxt = '/home/wyang/github/caffe/examples/lsp_patch_const/lsp-xianjie-deploy-classification.prototxt';
modelfile = '/home/wyang/Data/cache/caffe/LSP_P26_K17_patch/models/model-01-29/lsp-patch-train_iter_300000.caffemodel'

% init caffe network (spews logging info)
if exist('use_gpu', 'var')
  matcaffe_init(use_gpu);
else
  matcaffe_init(1, prototxt, modelfile, 1);
end


% prepare input

num_images = length(labels);
scores = zeros(dim,num_images,'single');

num_batches = ceil(num_images/batch_size)
initic=tic;
for bb = 1 : num_batches
    batchtic = tic;
    range = 1+batch_size*(bb-1):min(num_images,batch_size * bb);
    tic
    input_data = prepare_image_batch(data(:, :, :, range), batch_size);
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
% load('test_labels_fc6-conv5_200000.mat');
% scores = single(feats');
top_k = 1;
[s, idx] = sort(scores, 'descend');
accuracy = 0;

top_k_idx = idx(1:top_k, :)-1;
for i = 1:num_images
    if ~isempty(find(labels(i) == top_k_idx(:, i)))
        accuracy = accuracy + 1;
    end
end
accuracy = accuracy / num_images


% Top K Accuracy
load /home/wyang/github/caffe/examples/lsp_patch_const/extract_feature/jan-29-2015-mat/test_labels_fc6-conv5_300000.mat;
scores2 = single(feats');
[s2, idx2] = sort(scores2, 'descend');
accuracy2 = 0;

top_k_idx2 = idx2(1:top_k, :)-1;
for i = 1:num_images
    if ~isempty(find(labels(i) == top_k_idx2(:, i)))
        accuracy2 = accuracy2 + 1;
    end
end
accuracy2 = accuracy2 / num_images
% if exist('filename', 'var')
%     save([filename '.probs.mat'],'list_im','scores','-v7.3');
% end

% ------------------------------------------------------------------------
function images = prepare_image_batch(data,batch_size)
% ------------------------------------------------------------------------
num_images = size(data, 4);
if nargin < 2
    batch_size = num_images;
end

IMAGE_DIM = 28;

images = zeros(IMAGE_DIM,IMAGE_DIM,3,batch_size,'single');

for i=1:num_images
    % read file
    try
        im = data(:, :, :, i);
        % resize to fixed input size
        im = single(im);
%         im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
        % Transform GRAY to RGB
        if size(im,3) == 1
            im = cat(3,im,im,im);
        end
        
        %%!!!!!!!!!!!!!!!!!!
        % No need to permute from RGB to BGR
        
        % permute from RGB to BGR
%         im = im(:,:,[3 2 1]);
%         imshow(uint8(im + 100));pause;
%         im = permute(im, [2, 1, 3]);
%         imshow(uint8(im + 100));pause;
        images(:,:,:,i) = im;
    catch
        warning('Problems with file',image_files{i});
    end
end







