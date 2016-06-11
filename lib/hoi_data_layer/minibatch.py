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

import hoi_data_layer.spatial_relation as hdl_sr

def get_minibatch(roidb, num_classes, obj_hoi_int, ltype):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_obj_per_image = np.round(cfg.TRAIN.FG_OBJ_FRACTION * rois_per_image)
    fg_roi_per_image = np.round(cfg.TRAIN.FG_ROI_FRACTION * fg_obj_per_image)
    assert(cfg.TRAIN.USE_BG_OBJ or cfg.TRAIN.FG_OBJ_FRACTION == 1.0, \
        'FG_OBJ_FRACTION must be 1.0 if USE_BG_OBJ is false')

    # Initialize input image blobs, formatted for caffe
    if cfg.USE_CCL:
        im_blob_h = np.zeros((0, 3, 419, 419), dtype=np.float32)
        im_blob_o = np.zeros((0, 3, 419, 419), dtype=np.float32)
    else:
        im_blob_h = np.zeros((0, 3, 227, 227), dtype=np.float32)
        im_blob_o = np.zeros((0, 3, 227, 227), dtype=np.float32)
    if cfg.USE_SCENE:
        im_blob_s = np.zeros((0, 3, 227, 227), dtype=np.float32)
    if cfg.USE_SPATIAL == 1 or cfg.USE_SPATIAL == 2:
        # Interaction Patterns
        im_blob_sr = np.zeros((0, 2, 64, 64), dtype=np.float32)
    if cfg.USE_SPATIAL == 3 or cfg.USE_SPATIAL == 4:
        # 2D vector between box centers
        im_blob_sr = np.zeros((0, 2), dtype=np.float32)
    if cfg.USE_SPATIAL == 5 or cfg.USE_SPATIAL == 6:
        # Concat of box locations (x, y, w, h)
        im_blob_sr = np.zeros((0, 8), dtype=np.float32)
    if cfg.SHARE_O:
        score_o_blob = np.zeros((0, 1), dtype=np.float32)
    # if cfg.SHARE_V:
    #     # no additional blobs needed
    if cfg.USE_UNION:
        im_blob_ho = np.zeros((0, 3, 227, 227), dtype=np.float32)

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
        labels, im_rois, scores \
            = _sample_rois(roidb[im_i], fg_obj_per_image, fg_roi_per_image,
                           rois_per_image, num_classes, obj_hoi_int, ltype)

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
            if cfg.USE_SPATIAL > 0:
                if cfg.USE_SPATIAL == 1:
                    # do not keep aspect ratio
                    blob_sr, _, _ = hdl_sr.get_map_no_pad(box_h, box_o, 64)
                if cfg.USE_SPATIAL == 2:
                    # keep aspect ratio
                    blob_sr, _, _ = hdl_sr.get_map_pad(box_h, box_o, 64)
                if cfg.USE_SPATIAL == 3 or cfg.USE_SPATIAL == 5:
                    # do not keep aspect ratio
                    _, bxh_rs, bxo_rs = hdl_sr.get_map_no_pad(box_h, box_o, 64)
                if cfg.USE_SPATIAL == 4 or cfg.USE_SPATIAL == 6:
                    # keep aspect ratio
                    _, bxh_rs, bxo_rs = hdl_sr.get_map_pad(box_h, box_o, 64)
                if cfg.USE_SPATIAL == 3 or cfg.USE_SPATIAL == 4:
                    # 2D vector between box centers
                    cth = np.array([(bxh_rs[0] + bxh_rs[2])/2,
                                    (bxh_rs[1] + bxh_rs[3])/2])
                    cto = np.array([(bxo_rs[0] + bxo_rs[2])/2,
                                    (bxo_rs[1] + bxo_rs[3])/2])
                    blob_sr = cto - cth
                if cfg.USE_SPATIAL == 5 or cfg.USE_SPATIAL == 6:
                    # Concat of box locations (x, y, w, h)
                    bxh = np.array([(bxh_rs[0] + bxh_rs[2])/2,
                                    (bxh_rs[1] + bxh_rs[3])/2,
                                    bxh_rs[2] - bxh_rs[0],
                                    bxh_rs[3] - bxh_rs[1]])
                    bxo = np.array([(bxo_rs[0] + bxo_rs[2])/2,
                                    (bxo_rs[1] + bxo_rs[3])/2,
                                    bxo_rs[2] - bxo_rs[0],
                                    bxo_rs[3] - bxo_rs[1]])
                    blob_sr = np.hstack((bxh, bxo))
                im_blob_sr = np.vstack((im_blob_sr, blob_sr[None, :]))
            if cfg.SHARE_O:
                # Use natural log of object detection scores
                score_o = np.log(scores[i, 1])
                score_o_blob = np.vstack((score_o_blob, score_o))
            # if cfg.SHARE_V:
            #     # no additional blobs needed
            if cfg.USE_UNION:
                box_ho = _get_union_bbox(box_h, box_o)
                blob_ho = _get_one_blob(im, box_ho, 227, 227)
                im_blob_ho = np.vstack((im_blob_ho, blob_ho[None, :]))

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
    if cfg.USE_SPATIAL > 0:
        blobs['data_sr'] = im_blob_sr
    if cfg.SHARE_O:
        blobs['score_o'] = score_o_blob
    # if cfg.SHARE_V:
    #     # no additional blobs needed
    if cfg.USE_UNION:
        blobs = {'data_ho' : im_blob_ho, 'labels' : labels_blob}

    # if cfg.TRAIN.BBOX_REG:
    #     blobs['bbox_targets'] = bbox_targets_blob
    #     blobs['bbox_loss_weights'] = bbox_loss_blob

    return blobs

def _sample_rois(roidb, fg_obj_per_image, fg_roi_per_image, rois_per_image, 
                 num_classes, obj_hoi_int, ltype):
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

    if cfg.TRAIN.USE_BG_OBJ:
        # Compute number of background object RoIs to take from this image
        # (guarding against there being fewer than desired)
        bg_obj_per_this_image = \
            rois_per_image - fg_roi_per_this_image - bg_roi_per_this_image
        bg_obj_per_this_image = np.minimum(bg_obj_per_this_image,
                                           bg_obj_inds.shape[0])
        # Sample background object regions without replacement
        assert(bg_obj_inds.shape[0] > 0, \
            'Empty background object regions. Not expected!')
        bg_obj_smp_ind = npr.choice(range(bg_obj_inds.shape[0]),
                                    size=bg_obj_per_this_image, replace=False)
    else:
        # Do not include background object RoIs
        bg_obj_per_this_image = 0
        bg_obj_smp_ind = npr.choice(range(bg_obj_inds.shape[0]),
                                    size=0, replace=False)

    # Get sampled labels, boxes, detection scores
    # foreground RoIs
    lbls_fg_roi = np.zeros((fg_roi_per_this_image, num_classes), dtype='int32')
    rois_fg_roi = np.zeros((fg_roi_per_this_image, 8), dtype='uint16')
    scrs_fg_roi = np.zeros((fg_roi_per_this_image, 2), dtype='float32')
    for i, ind in enumerate(fg_roi_smp_ind):
        rid = fg_roi_inds[ind, 0]
        bid = fg_roi_inds[ind, 1]
        obj_id = roi_fg[rid]['obj_id']
        sid = obj_hoi_int[obj_id][0]
        eid = obj_hoi_int[obj_id][1]+1
        if ltype == 'SigmoidCrossEntropyLoss':
            lbls_fg_roi[i, sid:eid] = roi_fg[rid]['classes'][bid, :]
        if ltype == 'MultiLabelLoss':
            lbls_raw = roi_fg[rid]['classes'][bid, :]
            lbls_raw[lbls_raw == 0] = -1
            lbls_fg_roi[i, sid:eid] = lbls_raw
        rois_fg_roi[i, :] = roi_fg[rid]['boxes'][bid, :]
        scrs_fg_roi[i, :] = roi_fg[rid]['scores'][bid, :]
    # background RoIs
    lbls_bg_roi = np.zeros((bg_roi_per_this_image, num_classes), dtype='int32')
    rois_bg_roi = np.zeros((bg_roi_per_this_image, 8), dtype='uint16')
    scrs_bg_roi = np.zeros((bg_roi_per_this_image, 2), dtype='float32')
    for i, ind in enumerate(bg_roi_smp_ind):
        rid = bg_roi_inds[ind, 0]
        bid = bg_roi_inds[ind, 1]
        assert np.all(roi_fg[rid]['classes'][bid, :] == 0)
        # lbls_bg_roi[i, sid:eid] = roi_fg[rid]['classes'][bid, :]
        if ltype == 'MultiLabelLoss':
            obj_id = roi_fg[rid]['obj_id']
            sid = obj_hoi_int[obj_id][0]
            eid = obj_hoi_int[obj_id][1]+1
            lbls_bg_roi[i, sid:eid] = -1
        rois_bg_roi[i, :] = roi_fg[rid]['boxes'][bid, :]
        scrs_bg_roi[i, :] = roi_fg[rid]['scores'][bid, :]
    # background RoIs
    lbls_bg_obj = np.zeros((bg_obj_per_this_image, num_classes), dtype='int32')
    rois_bg_obj = np.zeros((bg_obj_per_this_image, 8), dtype='uint16')
    scrs_bg_obj = np.zeros((bg_obj_per_this_image, 2), dtype='float32')
    for i, ind in enumerate(bg_obj_smp_ind):
        rid = bg_obj_inds[ind, 0]
        bid = bg_obj_inds[ind, 1]
        if ltype == 'MultiLabelLoss':
            obj_id = roi_bg[rid]['obj_id']
            sid = obj_hoi_int[obj_id][0]
            eid = obj_hoi_int[obj_id][1]+1
            lbls_bg_obj[i, sid:eid] = -1
        rois_bg_obj[i, :] = roi_bg[rid]['boxes'][bid, :]
        scrs_bg_obj[i, :] = roi_bg[rid]['scores'][bid, :]
    # Stack arrays
    labels = np.vstack((lbls_fg_roi, lbls_bg_roi, lbls_bg_obj))
    rois = np.vstack((rois_fg_roi, rois_bg_roi, rois_bg_obj))
    scores = np.vstack((scrs_fg_roi, scrs_bg_roi, scrs_bg_obj))

    # bbox_targets, bbox_loss_weights = \
    #         _get_bbox_regression_labels(roidb['bbox_targets'][keep_inds, :],
    #                                     num_classes)

    # return labels, overlaps, rois, bbox_targets, bbox_loss_weights
    return labels, rois, scores

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

def _get_union_bbox(box1, box2):
    return np.array( \
        (np.minimum(box1[0], box2[0]), np.minimum(box1[1], box2[1]),
         np.maximum(box1[2], box2[2]), np.maximum(box1[3], box2[3])),
        dtype=np.uint16)

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
