# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

import numpy as np
from fast_rcnn.config import cfg
import utils.cython_bbox

def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        for j in xrange(len(roidb[i]['roi_fg'])):
            # need gt_overlaps as a dense array for argmax
            gt_overlaps = roidb[i]['roi_fg'][j]['gt_overlaps'].toarray()
            # max overlap with gt over classes (columns)
            # max_overlap is used for assigning background in get_minibatch()
            max_overlaps = gt_overlaps.max(axis=1)
            roidb[i]['roi_fg'][j]['max_overlaps'] = max_overlaps
            # gt binary class for multi-class classification
            classes = (gt_overlaps >= cfg.TRAIN.FG_THRESH).astype('int32')
            roidb[i]['roi_fg'][j]['classes'] = classes

# TODO: modify for multi-label
# def add_bbox_regression_targets(roidb):
#     """Add information needed to train bounding-box regressors."""
#     assert len(roidb) > 0
#     assert 'max_overlaps' in roidb[0], 'Did you call prepare_roidb first?'

#     num_images = len(roidb)
#     # Infer number of classes from the number of columns in gt_overlaps
#     num_classes = roidb[0]['gt_overlaps'].shape[1]
#     for im_i in xrange(num_images):
#         rois = roidb[im_i]['boxes']
#         max_overlaps = roidb[im_i]['max_overlaps']
#         max_classes = roidb[im_i]['max_classes']
#         roidb[im_i]['bbox_targets'] = \
#                 _compute_targets(rois, max_overlaps, max_classes)

#     # Compute values needed for means and stds
#     # var(x) = E(x^2) - E(x)^2
#     class_counts = np.zeros((num_classes, 1)) + cfg.EPS
#     sums = np.zeros((num_classes, 8))
#     squared_sums = np.zeros((num_classes, 8))
#     for im_i in xrange(num_images):
#         targets = roidb[im_i]['bbox_targets']
#         for cls in xrange(1, num_classes):
#             cls_inds = np.where(targets[:, 0] == cls)[0]
#             if cls_inds.size > 0:
#                 class_counts[cls] += cls_inds.size
#                 sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
#                 squared_sums[cls, :] += (targets[cls_inds, 1:] ** 2).sum(axis=0)

#     means = sums / class_counts
#     stds = np.sqrt(squared_sums / class_counts - means ** 2)

#     # Normalize targets
#     for im_i in xrange(num_images):
#         targets = roidb[im_i]['bbox_targets']
#         for cls in xrange(1, num_classes):
#             cls_inds = np.where(targets[:, 0] == cls)[0]
#             roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
#             roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]

#     # These values will be needed for making predictions
#     # (the predicts will need to be unnormalized and uncentered)
#     return means.ravel(), stds.ravel()

# TODO: modify for multi-label
# def _compute_targets(rois, overlaps, labels):
#     """Compute bounding-box regression targets for an image."""
#     # Ensure ROIs are floats
#     rois = rois.astype(np.float, copy=False)

#     # Indices of ground-truth ROIs
#     gt_inds = np.where(overlaps == 1)[0]
#     # Indices of examples for which we try to make predictions
#     ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

#     # Skip if gt_inds is empty
#     if gt_inds.size == 0:
#         return np.zeros((rois.shape[0], 9), dtype=np.float32)

#     # Get IoU overlap between each ex ROI and gt ROI
#     ex_gt_overlaps_h = utils.cython_bbox.bbox_overlaps(rois[ex_inds, 0:4],
#                                                        rois[gt_inds, 0:4])
#     ex_gt_overlaps_o = utils.cython_bbox.bbox_overlaps(rois[ex_inds, 4:8],
#                                                        rois[gt_inds, 4:8])
#     ex_gt_overlaps = (ex_gt_overlaps_h + ex_gt_overlaps_o) / 2
#     ex_gt_overlaps[ex_gt_overlaps_h == 0] = 0
#     ex_gt_overlaps[ex_gt_overlaps_o == 0] = 0

#     # Find which gt ROI each ex ROI has max overlap with:
#     # this will be the ex ROI's gt target
#     gt_assignment = ex_gt_overlaps.argmax(axis=1)
#     gt_rois = rois[gt_inds[gt_assignment], :]
#     ex_rois = rois[ex_inds, :]

#     targets = np.zeros((rois.shape[0], 9), dtype=np.float32)
#     dx_h, dy_h, dw_h, dh_h = _get_target_one_box(ex_rois[:, 0:4], 
#                                                  gt_rois[:, 0:4])
#     dx_o, dy_o, dw_o, dh_o = _get_target_one_box(ex_rois[:, 4:8], 
#                                                  gt_rois[:, 4:8])
#     targets[ex_inds, 0] = labels[ex_inds]
#     targets[ex_inds, 1] = dx_h
#     targets[ex_inds, 2] = dy_h
#     targets[ex_inds, 3] = dw_h
#     targets[ex_inds, 4] = dh_h
#     targets[ex_inds, 5] = dx_o
#     targets[ex_inds, 6] = dy_o
#     targets[ex_inds, 7] = dw_o
#     targets[ex_inds, 8] = dh_o
#     return targets

# def _get_target_one_box(ex_rois, gt_rois):
#     ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + cfg.EPS
#     ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + cfg.EPS
#     ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
#     ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

#     gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + cfg.EPS
#     gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + cfg.EPS
#     gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
#     gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

#     targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
#     targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
#     targets_dw = np.log(gt_widths / ex_widths)
#     targets_dh = np.log(gt_heights / ex_heights)

#     return targets_dx, targets_dy, targets_dw, targets_dh
