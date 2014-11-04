#!/usr/bin/env sh
# args for EXTRACT_FEATURE
TOOL=../../../build/tools
MODEL=/home/wyang/github/caffe/examples/lsp_pose/pose_caffenet_train_iter_25000.caffemodel 
PROTOTXT=/home/wyang/github/caffe/examples/lsp_pose/caffenet-pose-lsp-test.prototxt
# CONV 1
#LAYER=conv1
#LEVELDB=features_${LAYER}_0923

# FC8-POSE
LAYER=eltwise11
LEVELDB=train-03-nov

# LABEL
#LAYER=label
#LEVELDB=groundtruth_0310

rm -r -f $LEVELDB
BATCHSIZE=100

# args for LEVELDB to MAT
#DIM=290400 # conv1
#DIM=186624 # conv2
#DIM=64896 # conv3
DIM=28 # fc8-pose
OUT=$LEVELDB/test_labels.mat
BATCHNUM=10

$TOOL/extract_features.bin  $MODEL $PROTOTXT $LAYER $LEVELDB $BATCHNUM GPU
sudo python ../../../tools/wyang/leveldb2mat.py $LEVELDB $BATCHNUM  $BATCHSIZE $DIM $OUT 
