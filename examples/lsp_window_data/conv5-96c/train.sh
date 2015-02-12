#!/usr/bin/env sh

prefix=train-11-Feb-2015-conv5-96
postfix=train

mkdir cache
mkdir cache/$prefix
mkdir /home/wyang/Data/cache/caffe/LSP_P26_K17_patch/models/$prefix/

TOOLS=../../../build/tools

$TOOLS/caffe train -solver lsp-xianjie-solver.prototxt -gpu 2 2>&1 | tee cache/$prefix/$prefix-$postfix.log
