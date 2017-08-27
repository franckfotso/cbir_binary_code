# Project: cbir_binary_code
# File: config
# Written by: Romuald FOTSO
# Licensed: MIT License
# Copyright (c) 2017

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
cfg = __C

#
# General options
#
__C.GEN = edict()

#
# Database options
#
__C.DB = edict()

#
# Feature & binary code extraction options
#
__C.FE = edict()

# binary code length
__C.FE.BINARY_CODE_LEN = 48

# feature vector length
__C.FE.FEATURE_VECTOR_LEN = 4096

# model mean file
__C.FE.MODEL_MEAN = "data/imagenet/ilsvrc_2012_mean.npy"

# last layer name for binary code
__C.FE.MODEL_BIN_LAYER = "fc8_romyny_encode"

# last layer name for feature extraction
__C.FE.MODEL_FEAT_LAYER = "fc7"

# estimated image number
__C.FE.NUM_IM_ESTIMATED = 500

# max buffer size
__C.FE.MAX_BUF_SIZE = 1000 # 50000

# save binary code & features in mat file
__C.FE.SAVE_IN_MAT_FILE = 0

# binarycode file output
__C.FE.BIN_CODE_MAT_FILE = "output/binaryCode48.mat"

# deepfeature file output
__C.FE.DEEP_FEAT_MAT_FILE = "output/deepFeat48.mat"

# check dataset and remove wrong file
__C.FE.CHECK_DS = 1

#
# Training options
#
__C.TRAIN = edict()

#
# Test options
#
__C.TEST = edict()
