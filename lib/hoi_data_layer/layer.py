# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

HOIDataLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from hoi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
# from multiprocessing import Process, Queue

class HOIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
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
        # if cfg.TRAIN.USE_PREFETCH:
        #     return self._blob_queue.get()
        # else:
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._num_classes, self._obj_hoi_int,
                             self._ltype)

    def set_roidb(self, roidb, obj_hoi_int):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()
        self._obj_hoi_int = obj_hoi_int
        # if cfg.TRAIN.USE_PREFETCH:
        #     self._blob_queue = Queue(10)
        #     self._prefetch_process = BlobFetcher(self._blob_queue,
        #                                          self._roidb,
        #                                          self._num_classes)
        #     self._prefetch_process.start()
        #     # Terminate the child process when the parent exists
        #     def cleanup():
        #         print 'Terminating BlobFetcher'
        #         self._prefetch_process.terminate()
        #         self._prefetch_process.join()
        #     import atexit
        #     atexit.register(cleanup)

    def set_ltype(self, ltype):
        """Set the loss type for adjusting the labels."""
        assert ltype == 'SigmoidCrossEntropyLoss' or \
               ltype == 'MultiLabelLoss', \
               'currently only supports SigmoidCrossEntropyLoss ' \
               'and MultiLabelLoss for the classification loss.'
        self._ltype = ltype

    def setup(self, bottom, top):
        """Setup the HOIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {
            'data_h': 0,
            'data_o': 1}

        # data blob: holds a batch of N images, each with 3 channels
        if cfg.USE_CCL:
            top[0].reshape(1, 3, 419, 419)
            top[1].reshape(1, 3, 419, 419)
        else:
            top[0].reshape(1, 3, 227, 227)
            top[1].reshape(1, 3, 227, 227)

        if cfg.USE_SCENE:
            ind = len(self._name_to_top_map.keys())
            self._name_to_top_map['data_s'] = ind
            top[ind].reshape(1, 3, 227, 227)
        if cfg.USE_SPATIAL > 0:
            ind = len(self._name_to_top_map.keys())
            self._name_to_top_map['data_p'] = ind
            if cfg.USE_SPATIAL == 1 or cfg.USE_SPATIAL == 2:
                # Interaction Patterns
                top[ind].reshape(1, 2, 64, 64)
            if cfg.USE_SPATIAL == 3 or cfg.USE_SPATIAL == 4:
                # 2D vector between box centers
                top[ind].reshape(1, 2)
            if cfg.USE_SPATIAL == 5 or cfg.USE_SPATIAL == 6:
                # Concat of box locations (x, y, w, h)
                top[ind].reshape(1, 8)
        if cfg.SHARE_O:
            ind = len(self._name_to_top_map.keys())
            self._name_to_top_map['score_o'] = ind
            top[ind].reshape(1, 1)
        # if cfg.SHARE_V:
        #     # no additional inputs needed
        if cfg.USE_UNION:
            assert not cfg.USE_SCENE
            assert not cfg.USE_SPATIAL
            assert not cfg.SHARE_O
            assert not cfg.SHARE_V
            if cfg.USE_ROIPOOLING:
                # ROI Pooling
                self._name_to_top_map = {'data': 0, 'rois': 1}
            else:
                # crop
                self._name_to_top_map = {'data_ho': 0}

        # labels blob: R categorical binary labels in [0, ..., K-1]
        ind = len(self._name_to_top_map.keys())
        self._name_to_top_map['labels'] = ind
        top[ind].reshape(1, self._num_classes)

        # if cfg.TRAIN.BBOX_REG:
        #     self._name_to_top_map['bbox_targets'] = 4
        #     self._name_to_top_map['bbox_loss_weights'] = 5

        #     # bbox_targets blob: R bounding-box regression targets with 8
        #     # targets per class
        #     top[4].reshape(1, self._num_classes * 8)

        #     # bbox_loss_weights blob: At most 8 targets per roi are active;
        #     # thisbinary vector sepcifies the subset of active targets
        #     top[5].reshape(1, self._num_classes * 8)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

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

#     def _shuffle_roidb_inds(self):
#         """Randomly permute the training roidb."""
#         # TODO(rbg): remove duplicated code
#         self._perm = np.random.permutation(np.arange(len(self._roidb)))
#         self._cur = 0

#     def _get_next_minibatch_inds(self):
#         """Return the roidb indices for the next minibatch."""
#         # TODO(rbg): remove duplicated code
#         if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
#             self._shuffle_roidb_inds()

#         db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
#         self._cur += cfg.TRAIN.IMS_PER_BATCH
#         return db_inds

#     def run(self):
#         print 'BlobFetcher started'
#         while True:
#             db_inds = self._get_next_minibatch_inds()
#             minibatch_db = [self._roidb[i] for i in db_inds]
#             blobs = get_minibatch(minibatch_db, self._num_classes)
#             self._queue.put(blobs)
