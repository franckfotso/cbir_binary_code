# Project: cbir_binary_code
# File: feat_tools
# Written by: Romuald FOTSO
# Licensed: MIT License
# Copyright (c) 2017

import caffe
import numpy as np
import progressbar
import datetime
import math

def pycaffe_init_feat(use_gpu, model_prototxt, model_file):
    print ('model_prototxt: {}'.format(model_prototxt))
    print ('model_file: {}'.format(model_file))

    if use_gpu == 0:
        caffe.set_mode_cpu()
        print ('Using CPU Mode')
    else:
        caffe.set_mode_gpu()
        caffe.set_device(0)
        print ('Using GPU Mode')

    net = caffe.Net(str(model_prototxt), str(model_file), caffe.TEST)
    return net


def prepare_batch(image_files, transformer):
    l_im_transf = []
    for im_pn in image_files:
        try:
            im = caffe.io.load_image(im_pn)
            im_transf = transformer.preprocess('data', im)
            l_im_transf.append(im_transf)
        except:
            print ('prepare_batch> IOError: cannot identify image file - {}'.format(im_pn))

    return l_im_transf


def pycaffe_batch_feat(list_im, net, feat_len, ilsvrc_2012_mean_pn, layer_n):
    batch_size = 10  # RFM: init_val=10
    dim = feat_len
    for im_pn in list_im:
        #print im_pn
        pass

    # Adjust the batch size and dim to match with models/bvlc_reference_caffenet/deploy.prototxt
    if (len(list_im) % batch_size):
        print ('Assuming batches of {} images rest will be filled with zeros').format(batch_size)

    # init caffe network (spews logging info)
    # net = pycaffe_init_feat(use_gpu, model_prototxt, model_file)

    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(ilsvrc_2012_mean_pn)
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    # print 'mean-subtracted values:', zip('BGR', mu)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    # prepare input
    num_images = len(list_im)
    scores = np.zeros((num_images, dim), dtype="float32")
    num_batches = int(math.ceil(float(len(list_im)) / float(batch_size)))
    # print num_batches

    # monitor tasks
    step_title = layer_n + ', pycaffe_batch_feat: '
    widgets = [step_title, progressbar.Percentage(),
               " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=num_batches, widgets=widgets).start()

    # loop over all images
    for bb in range(0, num_batches):
        im_range = range(batch_size * (bb), min(num_images, batch_size * (bb + 1)))
        # print im_range
        startTime = datetime.datetime.now()
        sub_list_im = []
        for v in im_range:
            sub_list_im.append(list_im[v])
        # print sub_list_im
        l_im_transf = prepare_batch(sub_list_im, transformer)
        #         print ('Batch {} out of {} => {}% Done on {} seconds').format((bb+1),num_batches,\
        #             (float(bb+1)/float(num_batches))*100,(datetime.datetime.now() - startTime).total_seconds())

        #  with the default; we can also change it later, e.g., for different batch sizes)
        net.blobs['data'].reshape(len(im_range),  # batch size default(50)
                                  3,  # 3-channel (BGR) images
                                  227, 227)  # image size is 227x227
        net.blobs['data'].data[...] = l_im_transf

        # foward propagation
        output = net.forward()
        # the output probability vector for the batch
        output_prob = output[layer_n]
        scores[im_range] = output_prob
        pbar.update(bb)

    pbar.finish()

    return scores
