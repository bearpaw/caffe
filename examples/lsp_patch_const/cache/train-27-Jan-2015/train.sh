#!/usr/bin/env sh

prefix=train-27-Jan-2015
postfix=train

mkdir cache
mkdir cache/$prefix

TOOLS=../../build/tools

$TOOLS/caffe train -solver lsp-xianjie-solver.prototxt -gpu 1 2>&1 | tee cache/$prefix/$prefix-$postfix.log
