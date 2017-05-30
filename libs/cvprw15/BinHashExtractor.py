# Project: cbir_binary_code
# File: BinHashExtractor
# Written by: romyny
# Licensed: MIT License
# On: 29/05/17

from feat_tools import pycaffe_batch_feat
from feat_tools import pycaffe_init_feat
import numpy as np


class BinHashExtractor:

    def __init__(self, use_gpu, feat_len, model_prototxt, model_file,
                 ilsvrc_2012_mean_pn):
        print ('BinHashExtractor: init')
        self.use_gpu = use_gpu
        self.feat_len = feat_len
        self.model_prototxt = model_prototxt
        self.model_file = model_file
        self.ilsvrc_2012_mean_pn = ilsvrc_2012_mean_pn
        self.caffe_net = pycaffe_init_feat(use_gpu, model_prototxt, model_file)


    def extract(self, list_im, layer_n):
        print '-----------------------------------------'
        print '48-bits binary codes extraction of qry_im'
        print '-----------------------------------------'
        binary_codes = None
        if self.caffe_net != None:
            scores = pycaffe_batch_feat(list_im, self.caffe_net,
                                        self.feat_len, self.ilsvrc_2012_mean_pn, layer_n)
            binary_codes = np.where(scores >= 0.5, 1, 0)
        else:
            print 'BinHashExtractor>extract: Error, caffe_net is not defined'
        return binary_codes
