#!/usr/bin/env python

# Project: cbir_binary_code
# File: index_bin48_feat
# Written by: romyny
# Licensed: MIT License
# On: 29/05/17
# -------------------------------------
# USAGE:
# python tools/index_bin48_feat.py --use_gpu 1 \
# --model_file examples/foods25/RomynyNet_foods25_48_iter_30000.caffemodel \
# --feat_proto examples/main/deploy_fc7.prototxt \
# --bin_proto examples/foods25/RomynyNet_foods25_48_deploy.prototxt \
# --products_dir data/foods25/imgs \
# --deep_db data/foods25/foods25_48_deepDB.hdf5

import _init_paths
import argparse
from libs.cvprw15.BinHashExtractor import BinHashExtractor
from libs.cvprw15.DeepFeaturesExtractor import DeepFeaturesExtractor
from libs.config import *
from libs.indexer.DeepIndexer import DeepIndexer
import scipy.io as sio
from PIL import Image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--use_gpu", help='define GPU use',
                default=1, type=int)
ap.add_argument("-m", "--model_file", required=True,
                help="path to model file")
ap.add_argument("-f", "--feat_proto", required=True,
                help="path to features prototxt")
ap.add_argument("-b", "--bin_proto", required=True,
                help="path to binary hash code prototxt")
ap.add_argument("-t", "--products_dir", required=True,
                help="input data images list")
ap.add_argument("-d", "--deep_db", required=True,
                help="deepfeatures db")
args = vars(ap.parse_args())

use_gpu = 0
if args["use_gpu"]:
    use_gpu = 1

# initialize the deep feature indexer
di = DeepIndexer(args["deep_db"], estNumImages=cfg.FE.NUM_IM_ESTIMATED,
                 maxBufferSize=cfg.FE.MAX_BUF_SIZE, verbose=True)

'''
    48-bits binary codes extraction
'''
binHash_extr = BinHashExtractor(use_gpu, cfg.FE.BINARY_CODE_LEN, args["bin_proto"],
                                args["model_file"], cfg.FE.MODEL_MEAN)
list_im = []
products_DIR = args['products_dir']
l_product_categ = os.listdir(products_DIR)
for i, product_categ in enumerate(l_product_categ):
    product_categ_DIR = os.path.join(products_DIR, product_categ)
    l_product = os.listdir(product_categ_DIR)
    if cfg.FE.CHECK_DS == 1:
        print ('[{}/{}] Loading & checking product_categ: {}'.format(i + 1, len(l_product_categ), product_categ))

    for prod_fn in l_product:
        im_pn = os.path.join(product_categ_DIR, prod_fn)
        if not os.path.isdir(im_pn):
            try:
                if cfg.FE.CHECK_DS == 1:
                    im_test = Image.open(im_pn)
                    del im_test
                list_im.append(im_pn)
            except:
                os.remove(im_pn)
                print ('index_bin48_feat> IOError: cannot identify image file')
                print ('Wrong file deleted: {}'.format(im_pn))

binary_codes = binHash_extr.extract(list_im, cfg.FE.MODEL_BIN_LAYER)

# save binary codes
if cfg.FE.SAVE_IN_MAT_FILE == 1:
    sio.savemat(cfg.FE.BIN_CODE_MAT_FILE, mdict={'binary_codes': binary_codes, list_im: 'list_im'})

'''
    layer7 feature extraction
'''
df_extr = DeepFeaturesExtractor(use_gpu, cfg.FE.FEATURE_VECTOR_LEN, args["feat_proto"],
                                args["model_file"], cfg.FE.MODEL_MEAN)

df_l7 = df_extr.extract(list_im, cfg.FE.MODEL_FEAT_LAYER)

# save binary codes
if cfg.FE.SAVE_IN_MAT_FILE == 1:
    sio.savemat(cfg.FE.DEEP_FEAT_MAT_FILE, mdict={'feat_test': df_l7, 'list_im': list_im})

"""
    Indexing binary code & deep features in hdf5 system
"""
for i, (feature_vector, binary_code) in enumerate(zip(df_l7, binary_codes)):
    # check to see if progress should be displayed
    if i > 0 and i % 10 == 0:
        di._debug("saved {} images".format(i), msgType="[PROGRESS]")

    im_pn = list_im[i]
    filename = im_pn[im_pn.rfind(os.path.sep) + 1:]

    # index the features & binary code
    di.add(filename, feature_vector, binary_code)

# finish the indexing process
di.finish()

