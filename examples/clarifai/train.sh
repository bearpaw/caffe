#!/usr/bin/env sh

../../build/tools/caffe train --solver=solver.prototxt \
 --weights=/home/wyang/Data/models/caffe-model/clarifai_finetune_ilsvrc14_gt_sel_6per_1000_pretrain_box/pretrain_imagenet-clarifai_iter_780000
