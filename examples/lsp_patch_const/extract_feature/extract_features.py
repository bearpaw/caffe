#!/usr/bin/env python
import sys
caffe_root = '../../../'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'tools/wyang')

import subprocess
import leveldb2mat
import test

top_k = 10
# extract parameters
prototxt = "/home/wyang/github/caffe/examples/lsp_patch_const/lsp-xianjie-train-test.prototxt"
model_prefix = "/home/wyang/Data/cache/caffe/LSP_P26_K17_patch/models/model-01-29/lsp-patch-train_iter_"  # remember to change the model path
layer = "fc6-conv5"
leveldb = "jan-29-2015" # remember to change date
batchsize = 100
batchnum = top_k*10
dim = 183
out_dir = leveldb + "-mat/"
subprocess.call(["mkdir", out_dir])

# start iterations
model_idx = range(300000, 300001, 1000)
for idx in model_idx:
	print "Processing {}".format(idx);
	output = out_dir + "test_labels_" + layer + "_" + str(idx) +".mat"
	subprocess.call(["rm","-rf",leveldb])
	subprocess.call(['./'+caffe_root + 'build/tools/extract_features.bin',
            model_prefix+str(idx)+".caffemodel", prototxt, layer, leveldb, str(batchnum), 'GPU',
            str(1)])
	leveldb2mat.convert(leveldb, batchnum, batchsize, dim,  output)



