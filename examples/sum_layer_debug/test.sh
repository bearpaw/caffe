#!/usr/bin/env sh
postfix=`date +%m-%d`
tools=./build/tools
$tools/caffe test -gpu 1 -model ./examples/sum_layer_debug/lsp_test.prototxt \
 -weights /home/wyang/Data/Code/pose/chen_nips14_full/cache/lsp-03-06/lsp_iter_60000.caffemodel
