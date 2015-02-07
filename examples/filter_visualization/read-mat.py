#!/usr/bin/env python
import numpy as np
import scipy.io

mat = scipy.io.loadmat('/home/wyang/Data/Code/pose/PE1.41-milestone1/cache/LSP_K7/lsp-channel-const-mean.mat')
np.save('/home/wyang/github/caffe/examples/lsp_patch_const/lsp-channel-const-mean.npy', mat['meanimg'])