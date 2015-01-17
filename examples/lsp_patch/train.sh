#!/usr/bin/env sh

prefix=train-14-Jan-2015
postfix=train

mkdir cache
mkdir cache/$prefix

TOOLS=../../build/tools

$TOOLS/caffe train -solver lsp-xianjie-solver.prototxt -snapshot /home/wyang/Data/cache/caffe/LSP_P26_K17_patch/models/lsp-patch-train_iter_76000.solverstate -gpu 0 2>&1 | tee cache/$prefix/$prefix-$postfix.log
