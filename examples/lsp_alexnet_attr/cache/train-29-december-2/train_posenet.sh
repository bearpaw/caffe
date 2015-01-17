#!/usr/bin/env sh

prefix=train-29-december-2
postfix=left-arm-finetune-from-nov-12-lsp_elw

mkdir cache
mkdir cache/$prefix

TOOLS=../../build/tools

$TOOLS/caffe train -solver caffenet-pose-solver.prototxt -weights  /home/wyang/github/caffe-regression/examples/lsp_elw/pose_caffenet_train_iter_13000.caffemodel \
 -gpu 0 2>&1 | tee cache/$prefix/$prefix-$postfix.log
