# Project: cbir_binary_code
# File: dists
# Written by: Romuald FOTSO
# Licensed: MIT License
# Copyright (c) 2017

import numpy as np

def chi2_distance(histA, histB, eps=1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum(((np.array(histA, dtype='float32') - np.array(histB,  dtype='float32')) ** 2)
                     / (np.array(histA, dtype='float32') + np.array(histB, dtype='float32') + eps))

    # return the chi-squared distance
    return d
