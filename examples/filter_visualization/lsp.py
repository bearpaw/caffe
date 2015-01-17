#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import vis_square

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

net = caffe.Classifier(caffe_root + 'examples/lsp_patch/lsp-xianjie-deploy.prototxt',
                       '/home/wyang/Data/cache/caffe/LSP_P26_K17_patch/models/lsp-patch-train_iter_300000.caffemodel')

net.set_phase_test()
net.set_mode_gpu()
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
# net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
# net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
# net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

scores = net.predict([caffe.io.load_image('/home/wyang/Data/Code/pose/PE1.41-milestone1/cache/LSP_P26_K7/test/000001_1_1.jpg')])

[(k, v.data.shape) for k, v in net.blobs.items()]
[(k, v[0].data.shape) for k, v in net.params.items()]

# index four is the center crop
plt.imshow(net.deprocess('data', net.blobs['data'].data[4]))
plt.show()

# the parameters are a list of [weights, biases]

# Conv1
filters = net.params['conv1'][0].data
vis_square.vis_square(filters.transpose(0, 2, 3, 1))
print filters.shape
plt.show()

feat = net.blobs['conv1'].data[4, :]
vis_square.vis_square(feat, padval=1)
plt.show()

# Conv2
filters = net.params['conv2'][0].data
print filters.shape
vis_square.vis_square(filters[:32].reshape(32**2, 3, 3))
plt.show()

feat = net.blobs['conv2'].data[4, :36]
vis_square.vis_square(feat, padval=1)
plt.show()


# Conv3
filters = net.params['conv3'][0].data
print filters.shape
vis_square.vis_square(filters[:128].reshape(128**2, 3, 3))
plt.show()

feat = net.blobs['conv3'].data[4, :64]
vis_square.vis_square(feat, padval=1)
plt.show()

# Conv4
feat = net.blobs['conv4'].data[4]
vis_square.vis_square(feat, padval=1)
plt.show()

# Conv5
feat = net.blobs['conv5'].data[4]
vis_square.vis_square(feat, padval=1)
plt.show()