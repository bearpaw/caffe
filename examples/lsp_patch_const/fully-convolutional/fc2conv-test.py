#!/usr/bin/env python
# make sure that caffe is on the python path:
caffe_root = '../../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np
import matplotlib.pyplot as plt


conv_protofile = '../lsp-xianjie-deploy-full-conv.prototxt'
conv_modelfile = '../cache/lsp-patch-train-full-conv.caffemodel'

# load input and configure preprocessing
net_full_conv = caffe.Net(conv_protofile, conv_modelfile)
	
im = caffe.io.load_image(caffe_root+'matlab/testdata/pyra_01.jpg')
net_full_conv.set_phase_test()
np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy')
net_full_conv.set_mean('data', np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
net_full_conv.set_channel_swap('data', (2,1,0))
net_full_conv.set_raw_scale('data', 255.0)
# make classification map by forward and print prediction indices at each location
out = net_full_conv.forward_all(data=np.asarray([net_full_conv.preprocess('data', im)]))
print out['prob'][0].argmax(axis=0)
# show net input and confidence map (probability of the top prediction at each location)
plt.subplot(1, 2, 1)
plt.imshow(net_full_conv.deprocess('data', net_full_conv.blobs['data'].data[0]))
plt.subplot(1, 2, 2)
plt.imshow(out['prob'][0].max(axis=0))
plt.show()
