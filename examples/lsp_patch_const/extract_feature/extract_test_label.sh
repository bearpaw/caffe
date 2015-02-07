#!/usr/bin/env sh
# args for EXTRACT_FEATURE
TOOL=../../build/tools
MODEL=$1$5.caffemodel
PROTOTXT=$2
# CONV 1
#LAYER=conv1
#LEVELDB=features_${LAYER}_0923

# FC8-POSE
LAYER=$3
LEVELDB=$9/$4

# LABEL
#LAYER=label
#LEVELDB=groundtruth_0310

rm -r -f $LEVELDB
BATCHSIZE=$6

# args for LEVELDB to MAT
BATCHNUM=$7
DIM=$8 # fc8-pose
OUT=$LEVELDB-mat/test_labels_$LAYER_$5.mat

$TOOL/extract_features.bin  $MODEL $PROTOTXT $LAYER $LEVELDB $BATCHNUM GPU 0
python ../../tools/wyang/leveldb2mat.py $LEVELDB $BATCHNUM  $BATCHSIZE $DIM $OUT 
