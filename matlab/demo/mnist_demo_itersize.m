function mnist_demo_itersize
% function mnist_demo
addpath('../../matlab');
addpath('../../examples/mnist_matlab/mnistHelper');


caffe.reset_all();

% load data
train_images = loadMNISTImages('../../data/mnist/train-images-idx3-ubyte');
train_labels = loadMNISTLabels('../../data/mnist/train-labels-idx1-ubyte');
train_labels(train_labels==0) = 10; % Remap 0 to 10

% solver params
params.epoch = 10;
params.batch_size = 1;
params.iter_size = 64;
params.num = length(train_labels);
params.numCases = length(unique(train_labels));
params.channel = 1;
params.height = 28;
params.width = 28;
params.display = 6400;
 
solver_file = '../../examples/mnist_matlab/lenet_solver_itersize.prototxt';
model = '../../examples/mnist_matlab/lenet_itersize.prototxt';
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
    prob = solver.net.blobs('prob').get_data();

    [loss, diff] = mysoftmaxloss(double(prob), labels);
    accloss = accloss + loss;
    res = solver.net.backward({diff});
    res = res{1};
    
    wdiff = solver.net.params('ip2', 2).get_diff();
    bdiff = solver.net.blobs('ip1').get_diff();
    if mod(iter, params.iter_size) == 0
      solver.update();
      solver.net.reset_params();
    end

    if mod(iter, params.display) == 0
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
    prob = net.blobs('prob').get_data();
    [pred, I] = max(prob);
    pred_label = [pred_label I];
end

acc = mean(test_labels(:) == pred_label(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);

% -------------------------------------------------------------------------
function [loss, diff] = mysoftmaxloss(prob, label)
% -------------------------------------------------------------------------
numCases = length(label);
log_prob = log(prob);
loss = 0;
% groundTruth = full(sparse(label, 1:numCases, 1));
% % compute loss
% loss1 = -sum(sum((groundTruth.*log(prob))))/numCases;

for i = 1:numCases
  loss = loss - log_prob(label(i), i);
end
loss = loss / numCases;
% compute diff
diff = prob;
for i = 1:length(label) % for i-th training sample
  diff(label(i), i) =  (diff(label(i), i) - 1)/numCases;
end






