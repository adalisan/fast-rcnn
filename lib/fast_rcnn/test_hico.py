# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from utils.cython_nms import nms
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os

import scipy.io as sio
import h5py

def _get_4_side_bbox(bbox, im_width, im_height):
    assert(bbox.ndim == 1 and bbox.shape[0] == 4)
    # get radius
    w = bbox[2]-bbox[0]+1;
    h = bbox[3]-bbox[1]+1;
    r = (w+h)/2;
    # get boxes: indexes are zero-based
    bbox_l = np.array([np.maximum(bbox[0]-0.5*r,0),
                       bbox[1],
                       bbox[2]-0.5*w,
                       bbox[3]])
    bbox_t = np.array([bbox[0],
                       np.maximum(bbox[1]-0.5*r,0),
                       bbox[2],
                       bbox[3]-0.5*h])
    bbox_r = np.array([bbox[0]+0.5*w,
                       bbox[1],
                       np.minimum(bbox[2]+0.5*r,im_width-1),
                       bbox[3]])
    bbox_b = np.array([bbox[0],
                       bbox[1]+0.5*h,
                       bbox[2],
                       np.minimum(bbox[3]+0.5*r,im_height-1)])

    # return in the order left, top, right, bottom
    return bbox_l[None,:], bbox_t[None,:], bbox_r[None,:], bbox_b[None,:]

def _enlarge_bbox_ctx8(bbox, im_width, im_height):
    # get radius
    w = bbox[2]-bbox[0]+1;
    h = bbox[3]-bbox[1]+1;
    r = (w+h)/2
    # get
    bbox_en = np.array([np.maximum(bbox[0]-0.5*r,0),
                        np.maximum(bbox[1]-0.5*r,0),
                        np.minimum(bbox[2]+0.5*r,im_width-1),
                        np.minimum(bbox[3]+0.5*r,im_height-1)])
    return bbox_en[None,:]

def _enlarge_bbox(bbox, im_width, im_height):
    # TODO: change indexes to zero-based
    w = bbox[2] - bbox[0] + 1;
    h = bbox[3] - bbox[1] + 1;
    r = (w + h) / 2

    x_c = np.floor((bbox[2] + bbox[0]) / 2)
    y_c = np.floor((bbox[3] + bbox[1]) / 2)

    bboxes = np.array([[np.maximum(x_c - w / 2 - r, 1),
                    np.maximum(y_c - h / 2 - r, 1),
                    np.minimum(x_c + w / 2 + r, im_width),
                    np.minimum(y_c + h / 2 + r, im_height)]])

    return bboxes

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    
    # This script is only used for HICO (FLAG_HICO will always be False),
    # so we don't need to handle blob 'rois'
    for ind in xrange(cfg.TOP_K):
        if cfg.FEAT_TYPE == 4:
            im_rois = [0] * 4
            im_rois[0], im_rois[1], im_rois[2], im_rois[3], \
                = _get_4_side_bbox(rois[ind,:], im.shape[1], im.shape[0])
            for i, s in enumerate(['l','t','r','b']):
                key = 'rois_%d_%s' % (ind+1,s)
                blobs[key] = _get_rois_blob(im_rois[i], im_scale_factors)
        else:
            key = 'rois_%d' % (ind+1)
            if cfg.FLAG_ENLARGE:
                rois_e = _enlarge_bbox(rois[ind, :], im.shape[1], im.shape[0])
                blobs[key] = _get_rois_blob(rois_e, im_scale_factors)
            else:
                blobs[key] = _get_rois_blob(rois[ind:ind+1,:], im_scale_factors)

    return blobs, im_scale_factors

def _get_one_blob(im, bbox, len_w=None, len_h=None):
    # crop image
    # bbox indexes are zero-based
    im_focus = im[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
    im_focus = im_focus.astype(np.float32, copy=False)
    # subtract mean
    im_focus -= cfg.PIXEL_MEANS
    # scale
    if len_w is None and len_h is None:
        im_focus = cv2.resize(im_focus, (cfg.FOCUS_W, cfg.FOCUS_H),
                              interpolation=cv2.INTER_LINEAR)
    else:
        assert(len_w is not None and len_h is not None)
        im_focus = cv2.resize(im_focus, (len_w, len_h),
                              interpolation=cv2.INTER_LINEAR)
    # convert image to blob
    channel_swap = (2, 0, 1)
    im_focus = im_focus.transpose(channel_swap)
    return im_focus

def _get_blobs_focus(im, roidb):
    """Convert an image into network inputs for focus data layer."""
    blobs = {}

    # This script is only used for HICO (FLAG_HICO will always be False),
    # so we don't need to handle blob 'rois'
    for ind in xrange(cfg.TOP_K):  
        # Now we just take the top K detection bbox; should consider
        # sampling K bbox from a larger pool later
        if cfg.FLAG_TOP_THRESH:
            # we threshold the boxes by scores
            keep = np.where(roidb['scores'] >= cfg.TOP_THRESH)
            if keep[0].size == 0:
                xid = 1
            else:
                xid = np.amax(keep[0]) + 1
            pid = ind % xid
        else:
            pid = ind
        print pid,
        # adjust boxes by feature type
        if cfg.FEAT_TYPE == 4:
            h_org = im.shape[0]
            w_org = im.shape[1]
            box_l, box_t, box_r, box_b \
                = _get_4_side_bbox(roidb['boxes'][pid,:], w_org, h_org)
            # round box and convert type to uint16
            box_l = np.around(box_l[0,:]).astype(np.uint16)
            box_t = np.around(box_t[0,:]).astype(np.uint16)
            box_r = np.around(box_r[0,:]).astype(np.uint16)
            box_b = np.around(box_b[0,:]).astype(np.uint16)
            # save blob
            key = 'data_%d_l' % (ind+1)
            blobs[key] = _get_one_blob(im, box_l)[None, :]
            key = 'data_%d_t' % (ind+1)
            blobs[key] = _get_one_blob(im, box_t)[None, :]
            key = 'data_%d_r' % (ind+1)
            blobs[key] = _get_one_blob(im, box_r)[None, :]
            key = 'data_%d_b' % (ind+1)
            blobs[key] = _get_one_blob(im, box_b)[None, :]
        else:
            # save blob
            key = 'data_%d' % (ind+1)
            blobs[key] = _get_one_blob(im, roidb['boxes'][pid,:])[None, :]

    return blobs

def _get_blobs_focus_ho(im, rois_o, rois_h, im_base):
    """Convert an image into network inputs for focus data layer."""
    blobs = {}

    h_org = im.shape[0]
    w_org = im.shape[1]
    if cfg.MODE_OBJ == -1 and cfg.MODE_HMN == -1:
        for ind in xrange(cfg.OBJ_K):
            if cfg.FLAG_CTX8:
                boxes_o = rois_o[ind,:]
                bbox_en = _enlarge_bbox_ctx8(boxes_o, w_org, h_org)
                bbox_en = np.around(bbox_en[0,:]).astype(np.uint16)
                im_blob = np.zeros((1, 3, cfg.FOCUS_LEN_HO, cfg.FOCUS_LEN_HO),
                                   dtype=np.float32)
                im_blob[0, :, :, :] = _get_one_blob(im, bbox_en,
                                                    cfg.FOCUS_LEN_HO,
                                                    cfg.FOCUS_LEN_HO)
            else:
                im_blob = np.zeros((1, 3, cfg.FOCUS_H, cfg.FOCUS_W),
                                   dtype=np.float32)
                im_blob[0, :, :, :] = _get_one_blob(im, rois_o[ind,:])
            key = 'data_o%d' % (ind+1)
            blobs[key] = im_blob
        for ind in xrange(cfg.HMN_K):
            if cfg.FLAG_CTX8:
                boxes_h = rois_h[ind,:]
                bbox_en = _enlarge_bbox_ctx8(boxes_h, w_org, h_org)
                bbox_en = np.around(bbox_en[0,:]).astype(np.uint16)
                im_blob = np.zeros((1, 3, cfg.FOCUS_LEN_HO, cfg.FOCUS_LEN_HO),
                                   dtype=np.float32)
                im_blob[0, :, :, :] = _get_one_blob(im, bbox_en,
                                                    cfg.FOCUS_LEN_HO,
                                                    cfg.FOCUS_LEN_HO)
            else:
                im_blob = np.zeros((1, 3, cfg.FOCUS_H, cfg.FOCUS_W),
                                   dtype=np.float32)
                im_blob[0, :, :, :] = _get_one_blob(im, rois_h[ind,:])
            key = 'data_h%d' % (ind+1)
            blobs[key] = im_blob
    else:
        for ind in xrange(cfg.OBJ_K):
            if cfg.MODE_OBJ == 0:
                im_blob = np.zeros((1, 3, 227, 227), dtype=np.float32)
                im_blob[0, :, :, :] = _get_one_blob(im, rois_o[ind,:])
            if cfg.MODE_OBJ == 1 or cfg.MODE_OBJ == 2:
                boxes_o = rois_o[ind,:]
                bbox_en = _enlarge_bbox_ctx8(boxes_o, w_org, h_org)
                bbox_en = np.around(bbox_en[0,:]).astype(np.uint16)
                im_blob = np.zeros((1, 3, 419, 419), dtype=np.float32)
                im_blob[0, :, :, :] = _get_one_blob(im, bbox_en, 419, 419)
            key = 'data_o%d' % (ind+1)
            blobs[key] = im_blob
        for ind in xrange(cfg.HMN_K):
            if cfg.MODE_HMN == 0:
                im_blob = np.zeros((1, 3, 227, 227), dtype=np.float32)
                im_blob[0, :, :, :] = _get_one_blob(im, rois_h[ind,:])
            if cfg.MODE_HMN == 1 or cfg.MODE_HMN == 2:
                boxes_h = rois_h[ind,:]
                bbox_en = _enlarge_bbox_ctx8(boxes_h, w_org, h_org)
                bbox_en = np.around(bbox_en[0,:]).astype(np.uint16)
                im_blob = np.zeros((1, 3, 419, 419), dtype=np.float32)
                im_blob[0, :, :, :] = _get_one_blob(im, bbox_en, 419, 419)
            if cfg.MODE_HMN == 3:
                hmap_file = 'hmap_' + im_base.replace('.mat','.hdf5')
                hmap_file = 'caches/cache_pose_hmap/test2015/' + hmap_file
                f = h5py.File(hmap_file, 'r')
                im_blob = np.zeros((1, 16, 64, 64), dtype=np.float32)
                im_blob[0, :, :, :] = f['hmap'][:][ind,:]
            if cfg.MODE_HMN == 4:
                feat_file = 'hmap_' + im_base.replace('.mat','.hdf5')
                feat_file = 'caches/cache_pose_feat/test2015/' + feat_file
                f = h5py.File(feat_file, 'r')
                im_blob = np.zeros((1, 512, 64, 64), dtype=np.float32)
                im_blob[0, :, :, :] = f['feat'][:][ind,:]
            # assertion: yunfan added 1 pixel to all the coordinates ?
            if cfg.MODE_HMN == 3 or cfg.MODE_HMN == 4:
                boxes_h = f['det_keep'][:][ind,0:4]
                boxes_h = np.around(boxes_h)-1  # type float32
                boxes_cmp = roidb[im_i]['boxes_h'][ind,:].astype('float32')
                diff = np.abs(boxes_h - boxes_cmp)
                assert(np.all(diff <= 1))
                f.close()
            # TODO:
            # if cfg.MODE_HMN == 4:
            key = 'data_h%d' % (ind+1)
            blobs[key] = im_blob

    return blobs

def _get_blobs_use_cache(roidb):
    if cfg.FLAG_CTX8:
        ld_o = sio.loadmat(roidb['ctx_file_o'])
        ld_h = sio.loadmat(roidb['ctx_file_h'])
        feat_det_o  = ld_o['feat_det_pre_ctx']
        feat_det_h  = ld_h['feat_det_pre_ctx']
        boxes_det_o = ld_o['boxes_det']
        boxes_det_h = ld_h['boxes_det']
        key_base = 'pool6'
    else:
        ld_o = sio.loadmat(roidb['reg_file_o'])
        ld_h = sio.loadmat(roidb['reg_file_h'])
        feat_det_o  = ld_o['feat_det_pre_reg']
        feat_det_h  = ld_h['feat_det_pre_reg']
        boxes_det_o = ld_o['boxes_det']
        boxes_det_h = ld_h['boxes_det']
        key_base = 'fc6'
    if cfg.USE_FT == 0:
        feat_full_o = ld_o['feat_full_pre']
        feat_full_h = ld_h['feat_full_pre']
    if cfg.USE_FT == 1:
        feat_full_o = ld_o['feat_full_ftv']
        feat_full_h = ld_h['feat_full_ftv']
    if cfg.USE_FT == 2:
        feat_full_o = ld_o['feat_full_fto']
        feat_full_h = ld_h['feat_full_fto']
    # object det feature
    blobs = {}
    for ind in xrange(cfg.OBJ_K):
        assert(np.all(roidb['boxes_o'][ind,:] == boxes_det_o[ind,:]))
        key = key_base + '_o%d' % (ind+1)
        blobs[key] = feat_det_o[ind,:][None, :]
    # human det feature
    for ind in xrange(cfg.HMN_K):
        assert(np.all(roidb['boxes_h'][ind,:] == boxes_det_h[ind,:]))
        key = key_base + '_h%d' % (ind+1)
        blobs[key] = feat_det_h[ind,:][None, :]
    # full image feature
    if cfg.FLAG_FULLIM:
        assert(np.all(feat_full_o == feat_full_h))
        blobs['fc6_s'] = feat_full_o[None, :]
    return blobs

# def _bbox_pred(boxes, box_deltas):
#     """Transform the set of class-agnostic boxes into class-specific boxes
#     by applying the predicted offsets (box_deltas)
#     """
#     if boxes.shape[0] == 0:
#         return np.zeros((0, box_deltas.shape[1]))
# 
#     boxes = boxes.astype(np.float, copy=False)
#     widths = boxes[:, 2] - boxes[:, 0] + cfg.EPS
#     heights = boxes[:, 3] - boxes[:, 1] + cfg.EPS
#     ctr_x = boxes[:, 0] + 0.5 * widths
#     ctr_y = boxes[:, 1] + 0.5 * heights
# 
#     dx = box_deltas[:, 0::4]
#     dy = box_deltas[:, 1::4]
#     dw = box_deltas[:, 2::4]
#     dh = box_deltas[:, 3::4]
# 
#     pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
#     pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
#     pred_w = np.exp(dw) * widths[:, np.newaxis]
#     pred_h = np.exp(dh) * heights[:, np.newaxis]
# 
#     pred_boxes = np.zeros(box_deltas.shape)
#     # x1
#     pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
#     # y1
#     pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
#     # x2
#     pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
#     # y2
#     pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
# 
#     return pred_boxes

# def _clip_boxes(boxes, im_shape):
#     """Clip boxes to image boundaries."""
#     # x1 >= 0
#     boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
#     # y1 >= 0
#     boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
#     # x2 < im_shape[1]
#     boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
#     # y2 < im_shape[0]
#     boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
#     return boxes

def im_detect(net, im, roidb):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    if cfg.FLAG_FOCUS:
        if cfg.USE_CACHE:
            assert(cfg.FLAG_HO)
            blobs = _get_blobs_use_cache(roidb)
        else:
            if cfg.FLAG_HO:
                # TODO: add FLAG_TOP_THRESH
                # TODO: add feat4
                boxes_o = roidb['boxes_o']
                boxes_h = roidb['boxes_h']
                im_base = os.path.basename(roidb['reg_file_h'])
                blobs = _get_blobs_focus_ho(im, boxes_o, boxes_h, im_base)
            else:
                boxes = roidb['boxes']
                blobs = _get_blobs_focus(im, roidb)
            if cfg.FLAG_FULLIM:
                assert(cfg.FLAG_FOCUS == True)
                h_org = im.shape[0]
                w_org = im.shape[1]
                box_f = np.array((0,0,w_org-1,h_org-1),dtype='uint16')
                blobs['data_s'] = _get_one_blob(im, box_f)[None, :]
    else:
        blobs, unused_im_scale_factors = _get_blobs(im, roidb)

    # Disable box dedup for HICO
    # # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # # (some distinct image ROIs get mapped to the same feature ROI).
    # # Here, we identify duplicate feature ROIs, so we only compute features
    # # on the unique subset.
    # if cfg.DEDUP_BOXES > 0:
    #     v = np.array([1, 1e3, 1e6, 1e9, 1e12])
    #     hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
    #     _, index, inv_index = np.unique(hashes, return_index=True,
    #                                     return_inverse=True)
    #     blobs['rois'] = blobs['rois'][index, :]
    #     boxes = boxes[index, :]

    # reshape network inputs and concat input blobs
    if cfg.FLAG_FOCUS:
        if cfg.USE_CACHE:
            assert(cfg.FLAG_HO)
            if cfg.FLAG_CTX8:
                for ind in xrange(cfg.OBJ_K):
                    key = 'pool6_o%d' % (ind+1)
                    net.blobs[key].reshape(*(blobs[key].shape))
                for ind in xrange(cfg.HMN_K):
                    key = 'pool6_h%d' % (ind+1)
                    net.blobs[key].reshape(*(blobs[key].shape))
            else:
                for ind in xrange(cfg.OBJ_K):
                    key = 'fc6_o%d' % (ind+1)
                    net.blobs[key].reshape(*(blobs[key].shape))
                for ind in xrange(cfg.HMN_K):
                    key = 'fc6_h%d' % (ind+1)
                    net.blobs[key].reshape(*(blobs[key].shape))
            if cfg.FLAG_FULLIM:
                net.blobs['fc6_s'].reshape(*(blobs['fc6_s'].shape))
        else:
            if cfg.FLAG_HO:
                for ind in xrange(cfg.OBJ_K):
                    key = 'data_o%d' % (ind+1)
                    net.blobs[key].reshape(*(blobs[key].shape))
                for ind in xrange(cfg.HMN_K):
                    key = 'data_h%d' % (ind+1)
                    net.blobs[key].reshape(*(blobs[key].shape))
            else:
                for ind in xrange(cfg.TOP_K):
                    if cfg.FEAT_TYPE == 4:
                        for s in ['l','t','r','b']:
                            key = 'data_%d_%s' % (ind+1,s)
                            net.blobs[key].reshape(*(blobs[key].shape))
                    else:
                        key = 'data_%d' % (ind+1)
                        net.blobs[key].reshape(*(blobs[key].shape))
            if cfg.FLAG_FULLIM:
                net.blobs['data_s'].reshape(*(blobs['data_s'].shape))
    else:
        net.blobs['data'].reshape(*(blobs['data'].shape))
        for ind in xrange(cfg.TOP_K):
            if cfg.FEAT_TYPE == 4:
                for s in ['l','t','r','b']:
                    key = 'rois_%d_%s' % (ind+1,s)
                    net.blobs[key].reshape(*(blobs[key].shape))
            else:
                key = 'rois_%d' % (ind+1)
                net.blobs[key].reshape(*(blobs[key].shape))

    # forward pass    
    blobs_out = net.forward(**(blobs))

    # if cfg.TEST.SVM:
    #     # use the raw scores before softmax under the assumption they
    #     # were trained as linear SVMs
    #     scores = net.blobs['cls_score'].data
    # else:
    #     # use softmax estimated probabilities
    #     scores = blobs_out['cls_prob']
    scores = []
    
    # save feature
    if cfg.FLAG_FOCUS:
        if cfg.FLAG_HO:
            feats = net.blobs['score'].data
        else:
            feats = net.blobs['score'].data
    else:
        if cfg.FEAT_TYPE == 4 and not cfg.FLAG_SIGMOID:
            feats = net.blobs['fc7_concat'].data
        elif cfg.FLAG_SIGMOID:
            feats = net.blobs['cls_score'].data
        else:
            feats = net.blobs['fc7'].data

    # assert(cfg.TEST.BBOX_REG == False)
    # if cfg.TEST.BBOX_REG:
    #     # Apply bounding-box regression deltas
    #     box_deltas = blobs_out['bbox_pred']
    #     pred_boxes = _bbox_pred(boxes, box_deltas)
    #     pred_boxes = _clip_boxes(pred_boxes, im.shape)
    # else:
    #     # Simply repeat the boxes, once for each class
    #     pred_boxes = np.tile(boxes, (1, scores.shape[1]))
    pred_boxes = []

    # Disable box dedup for HICO
    # if cfg.DEDUP_BOXES > 0:
    #     # Map scores and predictions back to the original set of boxes
    #     scores = scores[inv_index, :]
    #     pred_boxes = pred_boxes[inv_index, :]
    #     feats = feats[inv_index, :]

    return scores, pred_boxes, feats

# def vis_detections(im, class_name, dets, thresh=0.3):
#     """Visual debugging of detections."""
#     import matplotlib.pyplot as plt
#     im = im[:, :, (2, 1, 0)]
#     for i in xrange(np.minimum(10, dets.shape[0])):
#         bbox = dets[i, :4]
#         score = dets[i, -1]
#         if score > thresh:
#             plt.cla()
#             plt.imshow(im)
#             plt.gca().add_patch(
#                 plt.Rectangle((bbox[0], bbox[1]),
#                               bbox[2] - bbox[0],
#                               bbox[3] - bbox[1], fill=False,
#                               edgecolor='g', linewidth=3)
#                 )
#             plt.title('{}  {:.3f}'.format(class_name, score))
#             plt.show()

# def apply_nms(all_boxes, thresh):
#     """Apply non-maximum suppression to all predicted boxes output by the
#     test_net method.
#     """
#     num_classes = len(all_boxes)
#     num_images = len(all_boxes[0])
#     nms_boxes = [[[] for _ in xrange(num_images)]
#                  for _ in xrange(num_classes)]
#     for cls_ind in xrange(num_classes):
#         for im_ind in xrange(num_images):
#             dets = all_boxes[cls_ind][im_ind]
#             if dets == []:
#                 continue
#             keep = nms(dets, thresh)
#             if len(keep) == 0:
#                 continue
#             nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
#     return nms_boxes

def test_net_hico(net, imdb, feat_root):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # # heuristic: keep an average of 40 detections per class per images prior
    # # to NMS
    # max_per_set = 40 * num_images
    # # heuristic: keep at most 100 detection per class per image prior to NMS
    # max_per_image = 100
    # # detection thresold for each class (this is adaptively set based on the
    # # max_per_set constraint)
    # thresh = -np.inf * np.ones(imdb.num_classes)
    # # top_scores will hold one minheap of scores per class (used to enforce
    # # the max_per_set constraint)
    # top_scores = [[] for _ in xrange(imdb.num_classes)]
    # # all detections are collected into:
    # #    all_boxes[cls][image] = N x 5 array of detections in
    # #    (x1, y1, x2, y2, score)
    # all_boxes = [[[] for _ in xrange(num_images)]
    #              for _ in xrange(imdb.num_classes)]

    # output_dir = get_output_dir(imdb, net)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    
    if len(feat_root) >= 4 and feat_root[-4:] == '.mat':
        # combine results in one mat file
        assert(cfg.FLAG_FOCUS and cfg.FLAG_HO)
        flag_comb = True
        feat_file = feat_root
        if os.path.isfile(feat_file):
            return
        if not os.path.exists(os.path.split(feat_file)[0]):
            os.makedirs(os.path.split(feat_file)[0])
        feats_comb = np.zeros((num_images, net.blobs['score'].data.shape[1]),
                              dtype=np.float32)
    else:
        # output one mat file for each image
        flag_comb = False
        if not os.path.exists(feat_root):
            os.makedirs(feat_root)

    assert(cfg.FLAG_HICO == True)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    roidb = imdb.roidb
    for i in xrange(num_images):
        im_name = os.path.splitext(os.path.basename(imdb.image_path_at(i)))[0]

        if not flag_comb:
            feat_file = os.path.join(feat_root, im_name + '.mat')
            if os.path.isfile(feat_file):
                continue

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes, feats = im_detect(net, im, roidb[i])
        _t['im_detect'].toc()
    
        # for j in xrange(1, imdb.num_classes):
        #     inds = np.where((scores[:, j] > thresh[j]) &
        #                     (roidb[i]['gt_classes'] == 0))[0]
        #     cls_scores = scores[inds, j]
        #     cls_boxes = boxes[inds, j*4:(j+1)*4]
        #     top_inds = np.argsort(-cls_scores)[:max_per_image]
        #     cls_scores = cls_scores[top_inds]
        #     cls_boxes = cls_boxes[top_inds, :]
        #     # push new scores onto the minheap
        #     for val in cls_scores:
        #         heapq.heappush(top_scores[j], val)
        #     # if we've collected more than the max number of detection,
        #     # then pop items off the minheap and update the class threshold
        #     if len(top_scores[j]) > max_per_set:
        #         while len(top_scores[j]) > max_per_set:
        #             heapq.heappop(top_scores[j])
        #         thresh[j] = top_scores[j][0]

        #     all_boxes[j][i] = \
        #             np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        #             .astype(np.float32, copy=False)

        #     if 0:
        #         keep = nms(all_boxes[j][i], 0.3)
        #         vis_detections(im, imdb.classes[j], all_boxes[j][i][keep, :])
        if not flag_comb:
            _t['misc'].tic()
            sio.savemat(feat_file, {'feat' : feats})
            _t['misc'].toc()
        else:
            feats_comb[i,:] = feats

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)
    
    if flag_comb:
        sio.savemat(feat_file, {'feat' : feats_comb})

    # for j in xrange(1, imdb.num_classes):
    #     for i in xrange(num_images):
    #         inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
    #         all_boxes[j][i] = all_boxes[j][i][inds, :]

    # # det_file = os.path.join(output_dir, 'detections.pkl')
    # with open(det_file, 'wb') as f:
    #     cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    # print 'Applying NMS to all detections'
    # nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)

    # print 'Evaluating detections'
    # imdb.evaluate_detections(nms_dets, output_dir)
