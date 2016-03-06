# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg

def get_minibatch(roidb, num_classes, obj_hoi_int):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_obj_per_image = np.round(cfg.TRAIN.FG_OBJ_FRACTION * rois_per_image)
    fg_roi_per_image = np.round(cfg.TRAIN.FG_ROI_FRACTION * fg_obj_per_image)

    # Initialize input image blobs, formatted for caffe
    if cfg.USE_CCL:
        im_blob_h = np.zeros((0, 3, 419, 419), dtype=np.float32)
        im_blob_o = np.zeros((0, 3, 419, 419), dtype=np.float32)
    else:
        im_blob_h = np.zeros((0, 3, 227, 227), dtype=np.float32)
        im_blob_o = np.zeros((0, 3, 227, 227), dtype=np.float32)
    if cfg.USE_SCENE:
        im_blob_s = np.zeros((0, 3, 227, 227), dtype=np.float32)

    # Now, build the region of interest and label blobs
    labels_blob = np.zeros((0, num_classes), dtype=np.float32)
    # bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
    # bbox_loss_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
    # all_overlaps = []
    for im_i in xrange(num_images):
        # read image
        im = cv2.imread(roidb[im_i]['image'])
        h_im = im.shape[0]
        w_im = im.shape[1]

        if roidb[im_i]['flipped']:
            im = im[:, ::-1, :]

        # labels, max_overlaps, im_rois, bbox_targets, bbox_loss \
        #     = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
        #                    num_classes)
        labels, im_rois \
            = _sample_rois(roidb[im_i], fg_obj_per_image, fg_roi_per_image,
                           rois_per_image, num_classes, obj_hoi_int)

        # Add to RoIs blob
        for i in xrange(labels.shape[0]):
            box_h = im_rois[i, 0:4]
            box_o = im_rois[i, 4:8]
            if cfg.USE_CCL:
                box_h = _enlarge_bbox_ccl(box_h.astype(np.float), w_im, h_im)
                box_o = _enlarge_bbox_ccl(box_o.astype(np.float), w_im, h_im)
                box_h = np.around(box_h).astype(np.uint16)
                box_o = np.around(box_o).astype(np.uint16)
                blob_h = _get_one_blob(im, box_h, 419, 419)
                blob_o = _get_one_blob(im, box_o, 419, 419)
            else:
                blob_h = _get_one_blob(im, box_h, 227, 227)
                blob_o = _get_one_blob(im, box_o, 227, 227)
            im_blob_h = np.vstack((im_blob_h, blob_h[None, :]))
            im_blob_o = np.vstack((im_blob_o, blob_o[None, :]))
            if cfg.USE_SCENE:
                box_s = np.array((0, 0, w_im-1, h_im-1), dtype='uint16')
                blob_s = _get_one_blob(im, box_s, 227, 227)
                im_blob_s = np.vstack((im_blob_s, blob_s[None, :]))

        # Add to labels, bbox targets, and bbox loss blobs
        labels_blob = np.vstack((labels_blob, labels))
        # bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
        # bbox_loss_blob = np.vstack((bbox_loss_blob, bbox_loss))
        # all_overlaps = np.hstack((all_overlaps, max_overlaps))

    # For debug visualizations
    # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)

    blobs = {'data_h': im_blob_h,
             'data_o': im_blob_o,
             'labels': labels_blob}

    if cfg.USE_SCENE:
        blobs['data_s'] = im_blob_s

    # if cfg.TRAIN.BBOX_REG:
    #     blobs['bbox_targets'] = bbox_targets_blob
    #     blobs['bbox_loss_weights'] = bbox_loss_blob

    return blobs

def _sample_rois(roidb, fg_obj_per_image, fg_roi_per_image, rois_per_image, 
                 num_classes, obj_hoi_int):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    roi_fg = roidb['roi_fg']
    roi_bg = roidb['roi_bg']

    # Initialize sampled indices
    fg_roi_inds = np.zeros((0, 2), dtype='int')
    bg_roi_inds = np.zeros((0, 2), dtype='int')
    bg_obj_inds = np.zeros((0, 2), dtype='int')

    # Select foreground RoIs as those with >= FG_THRESH overlap 
    #    and background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    for i in xrange(len(roi_fg)):
        max_overlaps = roi_fg[i]['max_overlaps']
        # foreground RoIs
        ind = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0][:, None]
        rid = i * np.ones((ind.shape[0], 1), dtype='int')
        fg_roi_inds = np.vstack((fg_roi_inds, np.hstack((rid, ind))))
        # background RoIs
        ind = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0][:, None]
        rid = i * np.ones((ind.shape[0], 1), dtype='int')
        bg_roi_inds = np.vstack((bg_roi_inds, np.hstack((rid, ind))))
    # Select background object RoIs
    for i in xrange(len(roi_bg)):
        ind = np.arange(roi_bg[i]['boxes'].shape[0])[:, None]
        rid = i * np.ones((roi_bg[i]['boxes'].shape[0], 1), dtype='int')
        bg_obj_inds = np.vstack((bg_obj_inds, np.hstack((rid, ind))))

    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_roi_per_this_image = np.minimum(fg_roi_per_image, fg_roi_inds.shape[0])
    # Sample foreground regions without replacement
    if fg_roi_inds.shape[0] > 0:
        fg_roi_smp_ind = npr.choice(range(fg_roi_inds.shape[0]),
                                    size=fg_roi_per_this_image, replace=False)
    else:
        fg_roi_smp_ind = np.zeros((0), dtype='int')
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_roi_per_this_image = fg_obj_per_image - fg_roi_per_this_image
    bg_roi_per_this_image = np.minimum(bg_roi_per_this_image,
                                       bg_roi_inds.shape[0])
    # Sample background regions without replacement
    if bg_roi_inds.shape[0] > 0:
        bg_roi_smp_ind = npr.choice(range(bg_roi_inds.shape[0]),
                                    size=bg_roi_per_this_image, replace=False)
    else:
        bg_roi_smp_ind = np.zeros((0), dtype='int')
    # Compute number of background object RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_obj_per_this_image = \
        rois_per_image - fg_roi_per_this_image - bg_roi_per_this_image
    bg_obj_per_this_image = np.minimum(bg_obj_per_this_image,
                                       bg_obj_inds.shape[0])
    # Sample background object regions without replacement
    if bg_obj_inds.shape[0] > 0:
        bg_obj_smp_ind = npr.choice(range(bg_obj_inds.shape[0]),
                                    size=bg_obj_per_this_image, replace=False)
    else:
        # Should not reach here
        raise Exception('Empty background object regions. Not expected!')

    # Get sampled labels and boxes
    # foreground RoIs
    lbls_fg_roi = np.zeros((fg_roi_per_this_image, num_classes), dtype='uint8')
    rois_fg_roi = np.zeros((fg_roi_per_this_image, 8), dtype='uint16')
    for i, ind in enumerate(fg_roi_smp_ind):
        rid = fg_roi_inds[ind, 0]
        bid = fg_roi_inds[ind, 1]
        obj_id = roi_fg[rid]['obj_id']
        sid = obj_hoi_int[obj_id][0]
        eid = obj_hoi_int[obj_id][1]+1
        lbls_fg_roi[i, sid:eid] = roi_fg[rid]['classes'][bid, :]
        rois_fg_roi[i, :] = roi_fg[rid]['boxes'][bid, :]
    # background RoIs
    lbls_bg_roi = np.zeros((bg_roi_per_this_image, num_classes), dtype='uint8')
    rois_bg_roi = np.zeros((bg_roi_per_this_image, 8), dtype='uint16')
    for i, ind in enumerate(bg_roi_smp_ind):
        rid = bg_roi_inds[ind, 0]
        bid = bg_roi_inds[ind, 1]
        obj_id = roi_fg[rid]['obj_id']
        sid = obj_hoi_int[obj_id][0]
        eid = obj_hoi_int[obj_id][1]+1
        lbls_bg_roi[i, sid:eid] = roi_fg[rid]['classes'][bid, :]
        rois_bg_roi[i, :] = roi_fg[rid]['boxes'][bid, :]
    # background RoIs
    lbls_bg_obj = np.zeros((bg_obj_per_this_image, num_classes), dtype='uint8')
    rois_bg_obj = np.zeros((bg_obj_per_this_image, 8), dtype='uint16')
    for i, ind in enumerate(bg_obj_smp_ind):
        rid = bg_obj_inds[ind, 0]
        bid = bg_obj_inds[ind, 1]
        rois_bg_obj[i, :] = roi_bg[rid]['boxes'][bid, :]
    # Stack arrays
    labels = np.vstack((lbls_fg_roi, lbls_bg_roi, lbls_bg_obj))
    rois = np.vstack((rois_fg_roi, rois_bg_roi, rois_bg_obj))

    # bbox_targets, bbox_loss_weights = \
    #         _get_bbox_regression_labels(roidb['bbox_targets'][keep_inds, :],
    #                                     num_classes)

    # return labels, overlaps, rois, bbox_targets, bbox_loss_weights
    return labels, rois

def _get_one_blob(im, bbox, w, h):
    # crop image
    # bbox indexes are zero-based
    im_trans = im[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
    im_trans = im_trans.astype(np.float32, copy=False)
    # subtract mean
    im_trans -= cfg.PIXEL_MEANS
    # scale
    im_trans = cv2.resize(im_trans, (w, h), interpolation=cv2.INTER_LINEAR)
    # convert image to blob
    channel_swap = (2, 0, 1)
    im_trans = im_trans.transpose(channel_swap)
    return im_trans

def _enlarge_bbox_ccl(bbox, w_im, h_im):
    # get radius
    w = bbox[2] - bbox[0] + 1;
    h = bbox[3] - bbox[1] + 1;
    r = (w + h) / 2
    # get enlarged bbox
    bbox_en = np.array([np.maximum(bbox[0] - 0.5 * r, 0),
                        np.maximum(bbox[1] - 0.5 * r, 0),
                        np.minimum(bbox[2] + 0.5 * r, w_im - 1),
                        np.minimum(bbox[3] + 0.5 * r, h_im - 1)])
    return bbox_en

# def _get_bbox_regression_labels(bbox_target_data, num_classes):
#     """Bounding-box regression targets are stored in a compact form in the
#     roidb.

#     This function expands those targets into the 4-of-4*K representation used
#     by the network (i.e. only one class has non-zero targets). The loss weights
#     are similarly expanded.

#     Returns:
#         bbox_target_data (ndarray): N x 4K blob of regression targets
#         bbox_loss_weights (ndarray): N x 4K blob of loss weights
#     """
#     clss = bbox_target_data[:, 0]
#     bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
#     bbox_loss_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
#     inds = np.where(clss > 0)[0]
#     for ind in inds:
#         cls = clss[ind]
#         start = 4 * cls
#         end = start + 4
#         bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
#         bbox_loss_weights[ind, start:end] = [1., 1., 1., 1.]
#     return bbox_targets, bbox_loss_weights

# def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
#     """Visualize a mini-batch for debugging."""
#     import matplotlib.pyplot as plt
#     for i in xrange(rois_blob.shape[0]):
#         rois = rois_blob[i, :]
#         im_ind = rois[0]
#         roi = rois[1:]
#         im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
#         im += cfg.PIXEL_MEANS
#         im = im[:, :, (2, 1, 0)]
#         im = im.astype(np.uint8)
#         cls = labels_blob[i]
#         plt.imshow(im)
#         print 'class: ', cls, ' overlap: ', overlaps[i]
#         plt.gca().add_patch(
#             plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
#                           roi[3] - roi[1], fill=False,
#                           edgecolor='r', linewidth=3)
#             )
#         plt.show()
