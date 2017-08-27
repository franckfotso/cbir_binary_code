# Project: cbir_binary_code
# File: _init_paths
# Written by: romyny
# Written by: Romuald FOTSO
# Licensed: MIT License
# Copyright (c) 2017

"""Set up paths for cbir_binary_code"""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '..', 'caffe', 'python')
add_path(caffe_path)

# Add libs to PYTHONPATH
libs_path = osp.join(this_dir, '..', 'libs')
#add_path(libs_path)

# Add project to PYTHONPATH
proj_path = osp.join(this_dir, '..')
add_path(proj_path)
