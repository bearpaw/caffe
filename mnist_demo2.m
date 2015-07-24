% function mnist_demo
addpath('./matlab');

weights = './examples/mnist/lenet_iter_10000.caffemodel';
solver_file = './examples/mnist_matlab_debug/lenet_solver.prototxt';
protofile = './examples/mnist_matlab_debug/lenet.prototxt';

gpu_id = 0;

caffe.set_mode_gpu();
caffe.set_device(gpu_id);

net = caffe.Net(protofile, weights, 'test'); % create net and load weights

% copy weight
convweight1 = squeeze( solver.net.params('conv1', 1).get_data());
solver.net.copy_from(weights);
convweight2 = squeeze( solver.net.params('conv1', 1).get_data());

net = solver.net;

data = rand(net.blobs('data').shape());
net.forward({data});

ip1data = net.blobs('ip1').get_data();
ip2data = net.blobs('ip2').get_data();

ip1w = net.params('ip1', 1).get_data();
ip2w = net.params('ip2', 1).get_data();

outdiff = rand(net.blobs('ip2').shape);
res = net.backward({outdiff});
data_diff = res{1};


ip1diff = net.blobs('ip1').get_diff();
ip2diff = net.blobs('ip2').get_diff();

a;





