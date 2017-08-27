# Project: cbir_binary_code
# File: DeepIndexer
# Written by: Romuald FOTSO
# Licensed: MIT License
# Copyright (c) 2017

from BaseIndexer import BaseIndexer
import numpy as np
import h5py

class DeepIndexer(BaseIndexer):
    def __init__(self, dbPath, estNumImages=500, maxBufferSize=50000, dbResizeFactor=2,
                 verbose=True):
        # call the parent constructor
        super(DeepIndexer, self).__init__(dbPath, estNumImages=estNumImages,
                                          maxBufferSize=maxBufferSize, dbResizeFactor=dbResizeFactor,
                                          verbose=verbose)

        # open the HDF5 database for writing and initialize the datasets within
        # the group
        self.db = h5py.File(self.dbPath, mode="w")
        self.imageIDDB = None
        self.indexDB = None
        self.deepfeaturesDB = None
        self.binarycodeDB = None

        # initialize the image IDs buffer, index buffer and the keypoints +
        # features buffer
        self.imageIDBuffer = []
        self.indexBuffer = []
        self.deepfeaturesBuffer = None
        self.binarycodeBuffer = []

        # initialize the total number of features in the buffer along with the
        # indexes dictionary
        self.totalFeatures = 0
        # 'index' for image_ids, index, & binarycode, 'deepfeatures' for features vectors
        self.idxs = {"index": 0, "deepfeatures": 0}

    def add(self, imageID, feature_vector, binary_code):
        # compute the starting and ending index for the features lookup
        start = self.idxs["deepfeatures"] + self.totalFeatures
        # end = start + len(feature_vector)
        end = start + 0  # 0 cause it is just one item added at time

        # update the image IDs buffer, features buffer, and index buffer,
        # followed by incrementing the feature count
        self.imageIDBuffer.append(imageID)
        self.indexBuffer.append((start, end))
        self.deepfeaturesBuffer = BaseIndexer.featureStack(np.hstack([feature_vector]),
                                                           self.deepfeaturesBuffer)
        # print "------------"
        # print (imageID)
        # print (binary_code)
        self.binarycodeBuffer.append(binary_code)
        # self.totalFeatures += len(feature_vector)
        self.totalFeatures += 1

        # check to see if we have reached the maximum buffer size
        if self.totalFeatures >= self.maxBufferSize:
            # if the databases have not been created yet, create them
            if None in (self.imageIDDB, self.indexDB, self.deepfeaturesDB, self.binarycodeDB):
                self._debug("initial buffer full")
                self._createDatasets()

            # write the buffers to file
            self._writeBuffers()

    def _createDatasets(self):
        # compute the average number of features extracted from the initial buffer
        # and use this number to determine the approximate number of features for
        # the entire dataset
        avgFeatures = self.totalFeatures / float(len(self.imageIDBuffer))
        approxFeatures = int(avgFeatures * self.estNumImages)

        # grab the feature vector size
        fvectorSize = self.deepfeaturesBuffer.shape[1]  # width
        bincodeSize = len(self.binarycodeBuffer[0])

        # initialize the datasets
        self._debug("creating datasets...")
        self.imageIDDB = self.db.create_dataset("image_ids", (self.estNumImages,),
                                                maxshape=(None,), dtype=h5py.special_dtype(vlen=unicode))
        self.indexDB = self.db.create_dataset("index", (self.estNumImages, 2),
                                              maxshape=(None, 2), dtype="int")
        self.deepfeaturesDB = self.db.create_dataset("deepfeatures",
                                                     (approxFeatures, fvectorSize), maxshape=(None, fvectorSize),
                                                     dtype="float")
        self.binarycodeDB = self.db.create_dataset("binarycode", (self.estNumImages, bincodeSize),
                                                   maxshape=(None, bincodeSize), dtype="int")

    def _writeBuffers(self):
        # write the buffers to disk
        self._writeBuffer(self.imageIDDB, "image_ids", self.imageIDBuffer, "index")
        self._writeBuffer(self.indexDB, "index", self.indexBuffer, "index")
        self._writeBuffer(self.deepfeaturesDB, "deepfeatures", self.deepfeaturesBuffer,
                          "deepfeatures")
        self._writeBuffer(self.binarycodeDB, "binarycode", self.binarycodeBuffer, "index")

        # increment the indexes
        self.idxs["index"] += len(self.imageIDBuffer)
        self.idxs["deepfeatures"] += self.totalFeatures

        # reset the buffers and feature counts
        self.imageIDBuffer = []
        self.indexBuffer = []
        self.deepfeaturesBuffer = None
        self.binarycodeBuffer = []
        self.totalFeatures = 0

    def finish(self):
        # if the databases have not been initialized, then the original
        # buffers were never filled up
        if None in (self.imageIDDB, self.indexDB, self.deepfeaturesDB, self.binarycodeDB):
            self._debug("minimum init buffer not reached", msgType="[WARN]")
            self._createDatasets()

        # write any unempty buffers to file
        self._debug("writing un-empty buffers...")
        self._writeBuffers()

        # compact datasets
        self._debug("compacting datasets...")
        self._resizeDataset(self.imageIDDB, "image_ids", finished=self.idxs["index"])
        self._resizeDataset(self.indexDB, "index", finished=self.idxs["index"])
        self._resizeDataset(self.deepfeaturesDB, "deepfeatures", finished=self.idxs["deepfeatures"])
        self._resizeDataset(self.binarycodeDB, "binarycode", finished=self.idxs["index"])

        # close the database
        self.db.close()
