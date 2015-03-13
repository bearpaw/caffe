#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import vis_square
import scipy.io

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

conv_protofile = '/home/wyang/github/caffe/examples/lsp_window_data/conv5-96c/lsp-xianjie-deploy-full-conv-36.prototxt'
conv_modelfile = '/home/wyang/github/caffe/examples/lsp_window_data/conv5-96c/lsp-patch-train-full-conv.caffemodel'

imagepath = '/home/wyang/Datasets/lsp_dataset/images/im1005.jpg'

# load input and configure preprocessing
net_full_conv = caffe.Net(conv_protofile, conv_modelfile)

# create image mean
imagemean = np.empty((3,160,70))
# R
for j in range(160):
    for k in range(70):
        imagemean[0][j][k]= 115
# G
for j in range(160):
    for k in range(70):
        imagemean[1][j][k]= 108
# B
for j in range(160):
    for k in range(70):
        imagemean[2][j][k]= 100
#print imagemean
	
# forward
im = caffe.io.load_image(imagepath)
#net_full_conv.set_phase_test()
net_full_conv.set_mean('data', imagemean)
net_full_conv.set_channel_swap('data', (2,1,0))
net_full_conv.set_raw_scale('data', 255.0)
# make classification map by forward and print prediction indices at each location
out = net_full_conv.forward_all(data=np.asarray([net_full_conv.preprocess('data', im)]))
print out['prob'][0].argmax(axis=0)
# show net_full_conv input and confidence map (probability of the top prediction at each location)
plt.subplot(1, 2, 1)
plt.imshow(net_full_conv.deprocess('data', net_full_conv.blobs['data'].data[0]))
plt.subplot(1, 2, 2)
plt.imshow(out['prob'][0].max(axis=0))
plt.show()
scipy.io.savemat('prob.mat', out);

#scores = net_full_conv.forward([caffe.io.load_image('/home/wyang/github/caffe/examples/filter_visualization/im0001.jpg')])

[(k, v.data.shape) for k, v in net_full_conv.blobs.items()]
[(k, v[0].data.shape) for k, v in net_full_conv.params.items()]

# index four is the center crop
plt.imshow(net_full_conv.deprocess('data', net_full_conv.blobs['data'].data[0]))
plt.show()

# the parameters are a list of [weights, biases]
params = dict({'conv1':net_full_conv.params['conv1'][0].data})
#print params['conv1']
scipy.io.savemat('params.mat', params);

# Conv1
filters = net_full_conv.params['conv1'][0].data
vis_square.vis_square(filters.transpose(0, 2, 3, 1))
print filters.shape
plt.show()

feat = net_full_conv.blobs['conv1'].data[0, :]
vis_square.vis_square(feat, padval=1)
plt.show()

# Conv2
filters = net_full_conv.params['conv2'][0].data
print filters.shape
vis_square.vis_square(filters[:96].reshape(96**2, 3, 3))
plt.show()

feat = net_full_conv.blobs['conv2'].data[0, :36]
vis_square.vis_square(feat, padval=1)
plt.show()


# Conv3
filters = net_full_conv.params['conv3'][0].data
print filters.shape
vis_square.vis_square(filters[:128].reshape(128**2, 3, 3))
plt.show()

feat = net_full_conv.blobs['conv3'].data[0, :64]
vis_square.vis_square(feat, padval=1)
plt.show()

# Conv4
filters = net_full_conv.params['conv3'][0].data
print filters.shape
vis_square.vis_square(filters[:128].reshape(128**2, 3, 3))
plt.show()

feat = net_full_conv.blobs['conv4'].data[0]
vis_square.vis_square(feat, padval=1)
plt.show()

# Conv5
filters = net_full_conv.params['conv3'][0].data
print filters.shape
vis_square.vis_square(filters[:128].reshape(128**2, 3, 3))
plt.show()

feat = net_full_conv.blobs['conv5'].data[0]
vis_square.vis_square(feat, padval=1)
plt.show()