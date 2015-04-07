#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/sum_layer_debug/lsp_solver.prototxt \
--weights /home/wyang/Data/Code/pose/chen_nips14_full/cache/lsp-03-06/fully_conv_net_by_net_surgery.caffemodel \
--gpu 1 2>&1 | tee aaaa.log
