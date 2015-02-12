#!/usr/bin/env python
# make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# original proto and model
protofile = '/home/wyang/github/caffe/examples/lsp_window_data/conv5-fg0.5/lsp-xianjie-deploy-predict.prototxt';
modelfile = '/home/wyang/Data/cache/caffe/LSP_P26_K17_patch/models/train-11-Feb-2015-conv5-fg0.5/lsp-patch-train_iter_22000.caffemodel';

# Load the original network and extract the fully-connected layers' parameters.
net = caffe.Net(protofile, modelfile)
params = ['fc6-conv5']
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

# Load the fully-convolutional network to transplant the parameters.
conv_protofile = '/home/wyang/github/caffe/examples/lsp_window_data/conv5-fg0.5/lsp-xianjie-deploy-full-conv.prototxt'
conv_modelfile = '/home/wyang/github/caffe/examples/lsp_window_data/conv5-fg0.5/lsp-patch-train-full-conv.caffemodel'

net_full_conv = caffe.Net(conv_protofile, modelfile)
params_full_conv = ['fc6-conv']
# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][1][...] = fc_params[pr][1]

for pr, pr_conv in zip(params, params_full_conv):
    out, in_, h, w = conv_params[pr_conv][0].shape
    W = fc_params[pr][0].reshape((out, in_, h, w))
    conv_params[pr_conv][0][...] = W

net_full_conv.save(conv_modelfile)
