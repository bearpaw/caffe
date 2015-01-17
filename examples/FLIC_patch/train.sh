#!/usr/bin/env sh

prefix=train-14-Jan-2015
postfix=train

mkdir cache
mkdir cache/$prefix

TOOLS=../../build/tools

$TOOLS/caffe train -solver flic-xianjie-solver.prototxt -weights /home/wyang/Data/cache/caffe/FLIC/two-labels/models/flic-patch-train_iter_120000.caffemodel -gpu 0 2>&1 | tee cache/$prefix/$prefix-$postfix.log
