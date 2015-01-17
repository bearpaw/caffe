#!/usr/bin/env python
import sys
caffe_root = '../../../'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'tools/wyang')

import subprocess
import leveldb2mat
import test


top_k = 5;
# extract parameters
prototxt = "/home/wyang/github/caffe/examples/lsp_alexnet/caffenet-pose-lsp-train-val-multable-fc.prototxt"
model_prefix = "/home/wyang/github/caffe/examples/lsp_alexnet/pose_caffenet_train_iter_"
layer = "elw9"
leveldb = "jan-06-2015"
batchsize = 100
batchnum = top_k*10
dim = 4
out_dir = leveldb + "-mat/"
subprocess.call(["mkdir", out_dir])

# start iterations
model_idx = range(10000, 12001, 1000)
for idx in model_idx:
	print "Processing {}".format(idx);
	output = out_dir + "test_labels_" + layer + "_" + str(idx) +".mat"
	subprocess.call(["rm","-rf",leveldb])
	subprocess.call(['./'+caffe_root + 'build/tools/extract_features.bin',
            model_prefix+str(idx)+".caffemodel", prototxt, layer, leveldb, str(batchnum), 'GPU',
            str(1)])
	leveldb2mat.convert(leveldb, batchnum, batchsize, dim,  output)



