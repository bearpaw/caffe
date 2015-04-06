function matcaffe_train_lenet
clear all;
clear mex;
addpath('mnistHelper');

use_gpu = 1;
model_def_file = 'lenet_deploy_no_softmat.prototxt';
model_file = '/home/wyang/iccv/caffe/examples/mnist/lenet_iter_1000.caffemodel';

caffe('reset');
caffe('init', model_def_file, model_file, 'train')

fprintf('Done with init\n');

% set to use GPU or CPU
if use_gpu
  fprintf('Using GPU Mode\n');
  caffe('set_mode_gpu');
else
  fprintf('Using CPU Mode\n');
  caffe('set_mode_cpu');
end
fprintf('Done with set_mode\n');



% load MNIST data
images = loadMNISTImages('../../data/mnist/train-images-idx3-ubyte');
labels = loadMNISTLabels('../../data/mnist/train-labels-idx1-ubyte');

% permulate data
idx = randperm(size(images, 2));
images = images(:, idx);
labels = labels(idx);

% Adjust the batch size and dim to match with models/bvlc_reference_caffenet/deploy.prototxt
batch_size = 64;
dim = 10;
scale = 0.00390625;

num_images = size(images, 2);
siz = sqrt(size(images, 1));
scores = zeros(dim,num_images,'single');
num_batches = ceil(num_images/batch_size)
initic=tic;

images = images*scale;

for bb = 1 : num_batches
    batchtic = tic;
    range = 1+batch_size*(bb-1):min(num_images,batch_size * bb);
    input_data = zeros(siz, siz, 1, batch_size, 'single');
    input_label = zeros(batch_size, 1);
    %%% prepare images
    for i = 1:length(range)
      im = reshape(images(:, range(i)), [siz, siz]);
%       im = im(:,:,[3 2 1]);
      input_data(:,:,:,i) = permute(im,[2 1 3]);    
      input_label(i) = labels(range(i));
    end
    loss1 =  caffe('forward', {input_data});
    loss1 = loss1{1};
    ip2 = squeeze(caffe('get_blob', 'ip2'));
    prob = mysoftmax(ip2);
    [loss, diff] = mysoftmaxloss(prob, input_label);
    %%% prepare top diff
    top_diff = ones(1, 1, 10, batch_size, 'single');
    for i = 1:batch_size
      top_diff(:, :, :, i) = diff(:, i);
    end
    b = caffe('backward', {top_diff});   
%     loss2 =  caffe('forward', {input_data});
%     loss2 = loss2{1};
%     isequal(loss1, loss2)
    fprintf('batch %d | loss = %.3f\n', bb, loss);
end
toc(initic);

%
function output = mysoftmax(input)
output = exp(input);
z = repmat(sum(output), [10, 1]);
output = output./z;

%
function [loss, diff] = mysoftmaxloss(prob, label)
label = label + 1; % index-0 to index-1
% compute loss
logp = log(prob);
loss = 0;
for i = 1:length(label)
  loss = loss + logp(label(i), i);
end
loss = -loss / size(prob, 2);

% compute diff
diff = prob;
for i = 1:length(label) % for i-th training sample
  diff(label(i), i) =  diff(label(i), i) - 1;
end
