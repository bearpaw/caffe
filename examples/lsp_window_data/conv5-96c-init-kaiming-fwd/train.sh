#!/usr/bin/env sh

prefix=train-04-March-2015-conv5-96-fwd
postfix=train

mkdir cache
mkdir cache/$prefix
mkdir /home/wyang/Data/cache/caffe/LSP_P26_K17_patch/models/$prefix/

TOOLS=../../../build/tools

$TOOLS/caffe train -solver lsp-xianjie-solver.prototxt -gpu 2 \
-snapshot /home/wyang/Data/cache/caffe/LSP_P26_K17_patch/models/train-04-March-2015-conv5-96-fwd/lsp-patch-train_iter_5000.solverstate 2>&1 | tee -a cache/$prefix/$prefix-$postfix.log
