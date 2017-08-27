#!/usr/bin/env python

# Project: cbir_binary_code
# File: deep_search
# Written by: Romuald FOTSO
# Licensed: MIT License
# Copyright (c) 2017
# -------------------------------------
# USAGE:
# python tools/deep_search.py --use_gpu 1 \
# --model_file examples/foods25/RomynyNet_foods25_48_iter_30000.caffemodel \
# --feat_proto examples/main/deploy_fc7.prototxt \
# --bin_proto examples/foods25/RomynyNet_foods25_48_deploy.prototxt \
# --deep_db data/foods25/foods25_48_deepDB.hdf5
# --query examples/foods25/imgs/koki/00152162.jpg

from __future__ import print_function
import _init_paths
import argparse, cv2, imutils
from libs.imr.DeepSearcher import DeepSearcher
from libs.ResultsMontage import ResultsMontage
from libs.cvprw15.BinHashExtractor import BinHashExtractor
from libs.cvprw15.DeepFeaturesExtractor import DeepFeaturesExtractor
from libs.imr import dists
from libs.config import *

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--use_gpu", help='define GPU use',
                default=1, type=int)
ap.add_argument("-m", "--model_file", required=True,
                help="path to model file")
ap.add_argument("-f", "--feat_proto", required=True,
                help="path to features prototxt")
ap.add_argument("-b", "--bin_proto", required=True,
                help="pathname of binary hash code prototxt")
ap.add_argument("-d", "--deep_db", required=True,
                help="Path to the features & binary code database")
ap.add_argument("-t", "--products_dir", required=True,
                help="input data images list")
ap.add_argument("-q", "--query", required=True,
                help="Path to the query image")
args = vars(ap.parse_args())

use_gpu = 0
if args["use_gpu"]:
    use_gpu = 1

queryRelevant = []

# grap image pathname
im_paths = {}
l_categs = os.listdir(args["products_dir"])
for categ_name in l_categs:
    l_imgs = os.listdir(args["products_dir"] + '/' + categ_name)
    for im_fn in l_imgs:
        im_paths[im_fn] = os.path.join(args["products_dir"] + '/' + categ_name, im_fn)

'''
    48-bits binary codes extraction
'''
binHash_extr = BinHashExtractor(use_gpu, cfg.FE.BINARY_CODE_LEN, args["bin_proto"],
                                args["model_file"], cfg.FE.MODEL_MEAN)

qry_binarycode = (binHash_extr.extract([args["query"]],
                                       cfg.FE.MODEL_BIN_LAYER))[0]  # 0 cause just 1 image
'''
    layer7 feature extraction
'''
df_extr = DeepFeaturesExtractor(use_gpu, cfg.FE.FEATURE_VECTOR_LEN, args["feat_proto"],
                                args["model_file"], cfg.FE.MODEL_MEAN)

qry_fVector = (df_extr.extract([args["query"]], cfg.FE.MODEL_FEAT_LAYER))[0]

dSearcher = DeepSearcher(args["deep_db"], distanceMetric=dists.chi2_distance)
# compute similarities
search_rlt = dSearcher.search(qry_binarycode, qry_fVector, numResults=20, maxCandidates=100)
print("[INFO] search took: {:.2f}s".format(search_rlt.search_time))

# initialize the results montage
montage = ResultsMontage((240, 320), 5, 20)

# load the query image and process it
queryImage = cv2.imread(args["query"])
cv2.imshow("Query", imutils.resize(queryImage, width=320))

# loop over the individual results
for (i, (score, resultID, resultIdx)) in enumerate(search_rlt.results):
    # load the result image and display it
    try:
        print("[RESULT] {result_num}. {result} - {score:.4f}".format(result_num=i + 1,
                                                                     result=resultID, score=score))

        result = cv2.imread("{}".format(im_paths[resultID]))
        montage.addResult(result, text="#{}".format(i + 1),
                          highlight=resultID in queryRelevant)
    except:
        print('Error: exception found on print')

# show the output image of results
cv2.imshow("Results", imutils.resize(montage.montage, height=700))

# save results
this_dir = osp.dirname(__file__)
out_rlts_DIR = os.path.join(this_dir,'..','output/results')
if not os.path.exists(out_rlts_DIR):
    os.mkdir(out_rlts_DIR)

qry_pn = args["query"]
out_rlts_file = os.path.join(out_rlts_DIR, qry_pn[qry_pn.rfind("/") + 1:])
cv2.imwrite(out_rlts_file, imutils.resize(montage.montage, height=700))

cv2.waitKey(0)
dSearcher.finish()
