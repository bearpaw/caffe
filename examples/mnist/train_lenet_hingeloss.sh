#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/mnist/lenet_hingeloss_solver.prototxt 2>&1 | tee mnist.log
