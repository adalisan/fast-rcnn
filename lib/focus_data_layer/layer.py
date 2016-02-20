# --------------------------------------------------------
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Yu-Wei Chao
# --------------------------------------------------------

"""FocusDataLayer implements a Caffe Python layer."""

import caffe
from fast_rcnn.config import cfg
from focus_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
# from multiprocessing import Process, Queue

import math
# from utils.timer import Timer

class FocusDataLayer(caffe.Layer):
    """Focus data layer."""

    def _shuffle_roidb_inds_obj(self):
        """Randomly permute the training roidb. Make sure each batch will have
        XXX examples with at least one positive label
        """
        # obj_per_batch = 2
        # bg_per_batch = 10
        obj_per_batch = int(round(cfg.TRAIN.IMS_PER_BATCH* cfg.FG_OBJ_FRACTION))
        bg_per_batch = cfg.TRAIN.IMS_PER_BATCH - obj_per_batch
        # print 'obj_per_batch: {}'.format(obj_per_batch)
        # print 'bg_per_batch: {}'.format(bg_per_batch)

        obj_id = [idx for idx, roi in enumerate(self._roidb)
                  if np.where(roi['label'] == 1)[0].size != 0];
        bg_id = list(set(range(len(self._roidb))) - set(obj_id))
        # print len(obj_id), len(bg_id)

        perm_obj = np.random.permutation(np.arange(len(obj_id)))
        perm_bg  = np.random.permutation(np.arange(len(bg_id)))
        # print perm_obj.size, perm_bg.size

        perm = np.array([],dtype='int64')
        num_batch = int(math.ceil(float(len(bg_id)) / float(bg_per_batch)))
        for ind in xrange(num_batch):
            ind_obj = range(ind*obj_per_batch,(ind+1)*obj_per_batch)
            ind_bg  = range(ind*bg_per_batch,(ind+1)*bg_per_batch)
            ind_obj = [i % len(obj_id) for i in ind_obj]
            ind_bg  = [i % len(bg_id) for i in ind_bg]
            perm = np.append(perm, perm_obj[ind_obj])
            perm = np.append(perm, perm_bg[ind_bg])

        self._perm = perm
        self._cur = 0

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        # self._perm = np.arange(len(self._roidb))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
        #     self._shuffle_roidb_inds()
        if self._fg_batch:
            # print 'shuffle roidb ind ... '
            if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._perm):
                self._shuffle_roidb_inds_obj()
        else:
            if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
                self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            # return self._blob_queue.get()
            assert 0, 'Should not reach here.'
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb, fg_batch):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        # self._shuffle_roidb_inds()
        if fg_batch:
            # print 'use fix fg obj num per batch'
            self._fg_batch = True
            self._shuffle_roidb_inds_obj()
        else:
            self._fg_batch = False
            self._shuffle_roidb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            # self._blob_queue = Queue(10)
            # self._prefetch_process = BlobFetcher(self._blob_queue,
            #                                      self._roidb,
            #                                      self._num_classes)
            # self._prefetch_process.start()
            # # Terminate the child process when the parent exists
            # def cleanup():
            #     print 'Terminating BlobFetcher'
            #     self._prefetch_process.terminate()
            #     self._prefetch_process.join()
            # import atexit
            # atexit.register(cleanup)
            assert 0, 'Should not reach here.'

    def setup(self, bottom, top):
        """Setup the FocusDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        # self._name_to_top_map = {
        #     'data': 0,
        #     'rois': 1,
        #     'labels': 2}
        # 
        # # data blob: holds a batch of N images, each with 3 channels
        # # The height and width (100 x 100) are dummy values
        # top[0].reshape(1, 3, 100, 100)
        # 
        # # rois blob: holds R regions of interest, each is a 5-tuple
        # # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # # rectangle (x1, y1, x2, y2)
        # top[1].reshape(1, 5)
        # 
        # # labels blob: R categorical labels in [0, ..., K] for K foreground
        # # classes plus background
        # top[2].reshape(1)

        # assertions
        assert(cfg.FLAG_HICO == True)
        assert(cfg.FLAG_FOCUS == True)
        assert(cfg.FLAG_SIGMOID == True)
        assert(cfg.TRAIN.BBOX_REG == False)
        
        self._name_to_top_map = {}
    
        if cfg.USE_CACHE:
            assert(cfg.FLAG_HO)
            for ind in xrange(0,cfg.OBJ_K):
                tind = ind
                if cfg.FLAG_CTX8:
                    key = 'pool6_o%d' % (ind+1)
                    self._name_to_top_map[key] = tind
                    top[tind].reshape(1, 4096, 3, 3)
                else:
                    key = 'fc6_o%d' % (ind+1)
                    self._name_to_top_map[key] = tind
                    top[tind].reshape(1, 4096)
            for ind in xrange(0,cfg.HMN_K):
                tind = ind + cfg.OBJ_K
                if cfg.FLAG_CTX8:
                    key = 'pool6_h%d' % (ind+1)
                    self._name_to_top_map[key] = tind
                    top[tind].reshape(1, 4096, 3, 3)
                else:
                    key = 'fc6_h%d' % (ind+1)
                    self._name_to_top_map[key] = tind
                    top[tind].reshape(1, 4096)
            if cfg.FLAG_FULLIM:
                ind = len(self._name_to_top_map.keys())
                self._name_to_top_map['fc6_s'] = ind
                top[ind].reshape(1, 4096)
        else:
            # data blob: holds a batch of N image tuples, each tuple contains
            # K cropped windows with 3 channels.
            if cfg.FLAG_HO:
                # TODO: add feat4
                if cfg.FLAG_CTX8:
                    LEN_H = cfg.FOCUS_LEN_HO
                    LEN_W = cfg.FOCUS_LEN_HO
                else:
                    LEN_H = cfg.FOCUS_H
                    LEN_W = cfg.FOCUS_W
                for ind in xrange(0,cfg.OBJ_K):
                    key = 'data_o%d' % (ind+1)
                    tind = ind
                    self._name_to_top_map[key] = tind
                    top[tind].reshape(1, 3, LEN_H, LEN_W)
                for ind in xrange(0,cfg.HMN_K):
                    key = 'data_h%d' % (ind+1)
                    tind = ind + cfg.OBJ_K
                    self._name_to_top_map[key] = tind
                    top[tind].reshape(1, 3, LEN_H, LEN_W)
            else:
                # TODO: add ctx8
                for ind in xrange(0,cfg.TOP_K):
                    if cfg.FEAT_TYPE == 4:
                        for i, s in enumerate(['l','t','r','b']):
                            # change _name_to_top_map
                            key = 'data_%d_%s' % (ind+1,s)
                            self._name_to_top_map[key] = ind*4+i
                            # reshape
                            top[ind*4+i].reshape(1, 3, cfg.FOCUS_H, cfg.FOCUS_W)
                    else:
                        # Note that key starts from 1 and _name_to_top_map[key] 
                        # starts from 0
                        key = 'data_%d' % (ind+1)
                        self._name_to_top_map[key] = ind
                        # The height and width (100 x 100) are dummy values
                        top[ind].reshape(1, 3, cfg.FOCUS_H, cfg.FOCUS_W)
            # full image feature
            if cfg.FLAG_FULLIM:
                ind = len(self._name_to_top_map.keys())
                self._name_to_top_map['data_s'] = ind
                top[ind].reshape(1, 3, cfg.FOCUS_H, cfg.FOCUS_W)

        # labels blob: binary categorical labels in [0, ..., K-1] for K 
        # foreground classes
        ind = len(self._name_to_top_map.keys())
        self._name_to_top_map['labels'] = ind
        top[ind].reshape(1, self._num_classes)

        # labels_vb blob
        if cfg.FLAG_SHARE_VB:
            ind = len(self._name_to_top_map.keys())
            self._name_to_top_map['labels_vb'] = ind
            top[ind].reshape(1, self._num_classes)

        # if cfg.TRAIN.BBOX_REG:
        #     self._name_to_top_map['bbox_targets'] = 3
        #     self._name_to_top_map['bbox_loss_weights'] = 4
        # 
        #     # bbox_targets blob: R bounding-box regression targets with 4
        #     # targets per class
        #     top[3].reshape(1, self._num_classes * 4)
        # 
        #     # bbox_loss_weights blob: At most 4 targets per roi are active;
        #     # thisbinary vector sepcifies the subset of active targets
        #     top[4].reshape(1, self._num_classes * 4)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        # timer = Timer()
        # timer.tic()
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)
        # print '  forward: ' + str(timer.toc())

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

# class BlobFetcher(Process):
#     """Experimental class for prefetching blobs in a separate process."""
#     def __init__(self, queue, roidb, num_classes):
#         super(BlobFetcher, self).__init__()
#         self._queue = queue
#         self._roidb = roidb
#         self._num_classes = num_classes
#         self._perm = None
#         self._cur = 0
#         self._shuffle_roidb_inds()
#         # fix the random seed for reproducibility
#         np.random.seed(cfg.RNG_SEED)
# 
#     def _shuffle_roidb_inds(self):
#         """Randomly permute the training roidb."""
#         # TODO(rbg): remove duplicated code
#         self._perm = np.random.permutation(np.arange(len(self._roidb)))
#         self._cur = 0
# 
#     def _get_next_minibatch_inds(self):
#         """Return the roidb indices for the next minibatch."""
#         # TODO(rbg): remove duplicated code
#         if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
#             self._shuffle_roidb_inds()
# 
#         db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
#         self._cur += cfg.TRAIN.IMS_PER_BATCH
#         return db_inds
#  
#     def run(self):
#         print 'BlobFetcher started'
#         while True:
#             db_inds = self._get_next_minibatch_inds()
#             minibatch_db = [self._roidb[i] for i in db_inds]
#             blobs = get_minibatch(minibatch_db, self._num_classes)
#             self._queue.put(blobs)
