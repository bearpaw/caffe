function mnist_demo_hingeloss
% function mnist_demo
addpath('../../matlab');
addpath('../../examples/mnist_matlab/mnistHelper');


caffe.reset_all();

% load data
train_images = loadMNISTImages('../../data/mnist/train-images-idx3-ubyte');
train_labels = loadMNISTLabels('../../data/mnist/train-labels-idx1-ubyte');
train_labels(train_labels==0) = 10; % Remap 0 to 10

% solver params
params.epoch = 15;
params.batch_size = 64;
params.num = length(train_labels);
params.numCases = length(unique(train_labels));
params.channel = 1;
params.height = 28;
params.width = 28;
params.display = 100;

solver_file = '../../examples/mnist_matlab/lenet_solver_hingeloss.prototxt';
model = '../../examples/mnist_matlab/lenet_hingeloss.prototxt';
weight = '../../examples/mnist_matlab/lenet.caffemodel';
gpu_id = 0;

caffe.set_mode_gpu();
caffe.set_device(gpu_id);

solver = caffe.Solver(solver_file);
iter = 0;

accloss = 0; % accumulated loss, for display
% Training model
for e = 1:params.epoch
  train_idx = randperm(params.num);
  disp('===================================');
  fprintf('Epoch # %d\n', e);
  for cur_idx = 1:params.batch_size:params.num-params.batch_size
    range = train_idx(cur_idx:cur_idx+params.batch_size-1);
    images = train_images(:, range);
    labels = train_labels(range);
    input_data = zeros(params.height, params.width, params.channel, params.batch_size, 'single');
    
    % prepare data
    for ii = 1:params.batch_size
      a = reshape(images(:, ii), [params.height, params.width]);
      input_data(:, :, 1, ii) = reshape(images(:, ii), [params.height, params.width]);
    end
    
    % train
    solver.net.forward({input_data}, false);
    prediction  = solver.net.blobs('ip2').get_data();
    
    [loss, diff] = myhingeloss(prediction, labels, 'L2');
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

% =========================================================================
% Testing
% =========================================================================
net = caffe.Net(model, weight, 'test'); % create testing net and load weights

test_images = loadMNISTImages('../../data/mnist/t10k-images-idx3-ubyte');
test_labels = loadMNISTLabels('../../data/mnist/t10k-labels-idx1-ubyte');
test_labels(test_labels==0) = 10; % Remap 0 to 10

test_batchsize = 100;

net.blobs('data').reshape([params.height, params.width, params.channel, test_batchsize]); % reshape blob 'data'
net.reshape();
pred_label = [];

for cur_idx = 1:test_batchsize:length(test_labels)
  range = [cur_idx:cur_idx+test_batchsize-1];
  images = test_images(:, range);
  labels = test_labels(range);
  input_data = zeros(params.height, params.width, params.channel, test_batchsize, 'single');
  
  % prepare data
  for ii = 1:test_batchsize
    input_data(:, :, 1, ii) = reshape(images(:, ii), [params.height, params.width]);
  end
  
  % train
  net.forward({input_data}, false);
  prob = net.blobs('ip2').get_data();
  [pred, I] = max(prob);
  pred_label = [pred_label I];
end

acc = mean(test_labels(:) == pred_label(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);

% -------------------------------------------------------------------------
function [loss, bottom_diff] = myhingeloss(predictions, label, norm)
% -------------------------------------------------------------------------
% reference: caffe hinge_loss_layer
if nargin < 4
  norm = 'L1';
end

if strcmp(norm, 'L1') == 0 && strcmp(norm, 'L2') == 0
  error('norm should be L1 or L2');
end

bottom_diff = predictions;

[dim, num] = size(predictions);
for i=1:num
  bottom_diff(label(i), i) = -1*bottom_diff(label(i), i);
end
for i=1:num
  for j=1:dim
    bottom_diff(j, i) = max(0, 1 + bottom_diff(j, i));
  end
end
loss = 0;
if strcmp(norm, 'L1')
  loss = sum(bottom_diff(:))/num;
else
  loss = sum(sum(bottom_diff.*bottom_diff))/num;
end
% compute gradient
for i=1:num
  bottom_diff(label(i), i) = -1*bottom_diff(label(i), i);
end
if strcmp(norm, 'L1')
  bottom_diff(bottom_diff > 0) = 1;
  bottom_diff(bottom_diff < 0) = -1;
  bottom_diff = bottom_diff / num;
else
  bottom_diff = bottom_diff*2/num;
end




