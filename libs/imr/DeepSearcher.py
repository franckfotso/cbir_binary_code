# Project: cbir_binary_code
# File: DeepSearcher
# Written by: romyny
# Licensed: MIT License
# On: 29/05/17

import datetime

import h5py
import numpy as np
from sklearn.metrics import hamming_loss
from sklearn.metrics.pairwise import pairwise_distances

from libs.imr import dists
from collections import namedtuple

class DeepSearcher:
    def __init__(self, deepDBPath, distanceMetric=dists.chi2_distance):
        # open both binary code and features vectors
        self.deepDB = h5py.File(deepDBPath, mode="r")
        self.num_df = 0

        """
        self.deepDB = None        
        self.d_deepDB = {}        
        for caffe_mdl in l_caffe_mdl:
            self.d_deepDB[caffe_mdl.name] = h5py.File(caffe_mdl.deepDB, mode="r")
            self.num_df += len(self.d_deepDB[caffe_mdl.name]['deepfeatures'])
        """

        # store distance metric selected
        self.distanceMetric = distanceMetric

    def search(self, qry_binarycode, qry_fVector, numResults=10, maxCandidates=200):
        # start the timer to track how long the search took
        startTime = datetime.datetime.now()

        # determine the candidates and sort them in ascending order so they can
        # be used to compare feature vector similarities
        l_candidates = self.findCandidates(qry_binarycode, maxCandidates)
        done_t1 = (datetime.datetime.now() - startTime).total_seconds()

        start_t2 = datetime.datetime.now()
        l_cand_fn, l_cand_id = [], []
        for (_, im_fn, im_id) in l_candidates:
            l_cand_fn.append(im_fn)
            l_cand_id.append(im_id)

        # grab feature vector of selected candidates
        l_image_ids = self.deepDB["image_ids"]
        l_cand_id = sorted(l_cand_id)
        l_can_fVector = self.deepDB["deepfeatures"][l_cand_id]

        results = {}
        for (can_id, can_fVector) in zip(l_cand_id, l_can_fVector):
            # compute distance between the two feature vector
            d = dists.chi2_distance(qry_fVector, can_fVector)
            d = float(d) / float(len(can_fVector))
            if (int)(d * 100) > 0:
                results[can_id] = d

        # sort all results such that small distance values are in the top
        results = sorted([(v, l_image_ids[k], k) for (k, v) in results.items()])
        results = results[:numResults]
        done_t2 = (datetime.datetime.now() - start_t2).total_seconds()

        print ("DeepSearcher.search: findcandidate_time on {} s".format(done_t1))
        print ("DeepSearcher.search: realsearch_time on {} s".format(done_t2))
        # return the search results
        SearchResult = namedtuple("SearchResult", ["results", "search_time"])
        return SearchResult(results, (datetime.datetime.now() - startTime).total_seconds())

    def findCandidates(self, qry_binarycode, maxCandidates):
        l_image_ids = self.deepDB["image_ids"]
        l_binarycode = self.deepDB["binarycode"]

        l_qry_bincode = [qry_binarycode]
        qry_D = pairwise_distances(np.array(l_qry_bincode), np.array(l_binarycode), 'hamming')[0]
        # get idx sorted in min order
        l_idx_sorted = qry_D.argsort()

        # sort HAMMING distance in ascending order
        maxCandidates = min(maxCandidates, len(l_binarycode))
        l_candidates = sorted([(qry_D[k], l_image_ids[k], k) for k in l_idx_sorted])
        l_candidates = l_candidates[:maxCandidates]

        return l_candidates

    def finish(self):
        pass
