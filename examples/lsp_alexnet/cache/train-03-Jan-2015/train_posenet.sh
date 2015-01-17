#!/usr/bin/env sh

prefix=train-03-Jan-2015
postfix=finetune

mkdir cache
mkdir cache/$prefix

TOOLS=../../build/tools

$TOOLS/caffe train -solver caffenet-pose-solver.prototxt \
 -weights /home/wyang/github/caffe/examples/lsp_alexnet/cache/train-02-Jan-2015-3/pose_caffenet_train_iter_75000.caffemodel \
 #-snapshot /home/wyang/github/caffe/examples/lsp_alexnet/cache/train-02-Jan-2015-3/pose_caffenet_train_iter_75000.solverstate \
 -gpu 0 2>&1 | tee cache/$prefix/$prefix-$postfix.log
