function mnist_demo_hingeloss_map
% function mnist_demo
addpath('../../matlab');
addpath('../../examples/mnist_matlab/mnistHelper');


caffe.reset_all();

% load data
train_images = loadMNISTImages('../../data/mnist/train-images-idx3-ubyte');
train_labels = loadMNISTLabels('../../data/mnist/train-labels-idx1-ubyte');
train_labels(train_labels==0) = 10; % Remap 0 to 10

% solver params
params.epoch = 30;
params.map_size = 64; % batch_size = 64/4 = 16
params.num = length(train_labels);
params.numCases = length(unique(train_labels));
params.channel = 1;
params.height = 28;
params.width = 28;
params.display = 100;

solver_file = '../../examples/mnist_matlab/lenet_solver_hingeloss_conv.prototxt';
model = '../../examples/mnist_matlab/lenet_hingeloss_conv.prototxt';
weight = '../../examples/mnist_matlab/lenet_hingeloss_conv.caffemodel';
gpu_id = 0;

caffe.set_mode_gpu();
caffe.set_device(gpu_id);

% =========================================================================
% Training
% =========================================================================
% if ~exist(weight, 'file')
  solver = caffe.Solver(solver_file);
  iter = 0;
  
  accloss = 0; % accumulated loss, for display
  % Training model
  for e = 1:params.epoch
    train_idx = randperm(params.num);
    disp('===================================');
    fprintf('Epoch # %d\n', e);
    for cur_idx = 1:params.map_size:params.num-params.map_size
      range = train_idx(cur_idx:cur_idx+params.map_size-1);
      raw_images = train_images(:, range);
      raw_labels = train_labels(range);
      input_data = zeros(params.height*2, params.width*2, params.channel, params.map_size/4, 'single');
      labels = zeros(2, 2,  params.map_size/4);
      % prepare data
      c = 1;
      for ii = 1:4:params.map_size
        im1 = reshape(raw_images(:, ii), [params.height, params.width]);
        im2 = reshape(raw_images(:, ii+1), [params.height, params.width]);
        im3 = reshape(raw_images(:, ii+2), [params.height, params.width]);
        im4 = reshape(raw_images(:, ii+3), [params.height, params.width]);
        
        im = [im1 im2; im3 im4];
        labels(:, :, c) = [raw_labels(ii), raw_labels(ii+1); ...
          raw_labels(ii+2), raw_labels(ii+3)];
        input_data(:, :, 1, c) = im;
        c = c+1;
      end
      
      %     for ii = 1:16
      %       imagesc(input_data(:, :, :, ii));
      %       title(fprintf('%d, %d\n%d, %d\n', labels(1, 1, ii), labels(1, 2, ii),...
      %       labels(2, 1, ii), labels(2, 2, ii)));
      %       pause;
      %     end
      
      % train
      solver.net.forward({input_data}, false);
      prediction  = solver.net.blobs('ip2').get_data();
      
      labels = uint8(imresize(labels, 4, 'nearest'));
      [loss, diff] = myhingeloss(prediction, labels, 'L1');
      accloss = accloss + loss;
      %         solver.net.reset_params(); % seems no use
      res = solver.net.backward({diff});
      res = res{1};
      solver.update();
      
      if iter == 0
        fprintf('Iter %6d: Loss %.6f\n', iter, accloss);
        accloss = 0;
      elseif mod(iter, params.display) == 0
        fprintf('Iter %6d: Loss %.6f\n', iter, accloss/params.display);
        accloss = 0;
      end
      iter = iter + 1;
    end
  end
  
  % Save model
  solver.net.save(weight);
% end
% =========================================================================
% Testing
% =========================================================================
net = caffe.Net(model, weight, 'test'); % create testing net and load weights

test_images = loadMNISTImages('../../data/mnist/t10k-images-idx3-ubyte');
test_labels = loadMNISTLabels('../../data/mnist/t10k-labels-idx1-ubyte');
test_labels(test_labels==0) = 10; % Remap 0 to 10

test_batchsize = 100;

net.blobs('data').reshape([params.height*2, params.width*2, params.channel, test_batchsize/4]); % reshape blob 'data'
net.reshape();
raw_pred_label = [];
acc = [];

for cur_idx = 1:test_batchsize:length(test_labels)
  range = [cur_idx:cur_idx+test_batchsize-1];
  raw_images = train_images(:, range);
  raw_labels = train_labels(range);
  input_data = zeros(params.height*2, params.width*2, params.channel, test_batchsize/4, 'single');
  labels = zeros(2, 2,  test_batchsize/4);
  % prepare data
  c = 1;
  for ii = 1:4:test_batchsize
    im1 = reshape(raw_images(:, ii), [params.height, params.width]);
    im2 = reshape(raw_images(:, ii+1), [params.height, params.width]);
    im3 = reshape(raw_images(:, ii+2), [params.height, params.width]);
    im4 = reshape(raw_images(:, ii+3), [params.height, params.width]);
    
    im = [im1 im2; im3 im4];
    labels(:, :, c) = [raw_labels(ii), raw_labels(ii+1); ...
      raw_labels(ii+2), raw_labels(ii+3)];
    input_data(:, :, 1, c) = im;
    c = c+1;
  end
  
  % test
  net.forward({input_data}, false);
  prob = net.blobs('ip2').get_data();
  [~, raw_pred_label] = max(prob, [], 3);
  raw_pred_label = squeeze(raw_pred_label);
  pred_label = zeros(2, 2, test_batchsize/4);
  for n = 1:test_batchsize/4
    pred_label(1, 1, n) = max_count_ele(raw_pred_label(1:4, 1:4, n));
    pred_label(1, 2, n) = max_count_ele(raw_pred_label(1:4, 5:8, n));
    pred_label(2, 1, n) = max_count_ele(raw_pred_label(5:8, 1:4, n));
    pred_label(2, 2, n) = max_count_ele(raw_pred_label(5:8, 5:8, n));
  end
  
%   for ii = 1:25
%     imagesc(input_data(:, :, :, ii)); hold on;
%     title(sprintf('%d, %d\n%d, %d\n', pred_label(1, 1, ii), pred_label(1, 2, ii),...
%       pred_label(2, 1, ii), pred_label(2, 2, ii)));
%     pause;
%   end
  acc = [acc mean(pred_label(:) == labels(:))]; 
end
acc = mean(acc);
fprintf('Accuracy: %0.3f%%\n', acc * 100);

% -------------------------------------------------------------------------
function [loss, bottom_diff] = myhingeloss(predictions, label, norm)
% -------------------------------------------------------------------------
if nargin < 4
  norm = 'L1';
end

if strcmp(norm, 'L1') == 0 && strcmp(norm, 'L2') == 0
  error('norm should be L1 or L2');
end
[width, height, dim, num] = size(predictions);
bottom_diff = predictions;

for i = 1:num
  for h = 1:height
    for w = 1:width
      bottom_diff(w, h, label(w, h, i), i) = -1*bottom_diff(w, h, label(w, h, i), i);
    end
  end
end
for i=1:num
  for j=1:dim
    bottom_diff(:, :, j, i) = max(0, 1 + bottom_diff(:, :, j, i));
  end
end
loss = 0;
if strcmp(norm, 'L1')
  loss = sum(bottom_diff(:))/(num*width*height);
else
  loss = sum(sum(bottom_diff.*bottom_diff))/(num*width*height);
end
% compute gradient
for i = 1:num
  for h = 1:height
    for w = 1:width
      bottom_diff(w, h, label(w, h, i), i) = -1*bottom_diff(w, h, label(w, h, i), i);
    end
  end
end
if strcmp(norm, 'L1')
  bottom_diff(bottom_diff > 0) = 1;
  bottom_diff(bottom_diff < 0) = -1;
  bottom_diff = bottom_diff / (num*width*height);
else
  bottom_diff = bottom_diff*2/(num*width*height);
end

% -------------------------------------------------------------------------
function maxval = max_count_ele(M)
% -------------------------------------------------------------------------
table = tabulate(M(:));
[~, idx] = max(table(:, 2));
maxval = table(idx);

