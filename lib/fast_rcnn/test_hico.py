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
import hoi_data_layer.spatial_relation as hdl_sr

def _get_union_bbox(box1, box2):
    return np.array( \
        (np.minimum(box1[0], box2[0]), np.minimum(box1[1], box2[1]),
         np.maximum(box1[2], box2[2]), np.maximum(box1[3], box2[3])),
        dtype=np.uint16)

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

# def _bbox_pred(boxes, box_deltas):
#     """Transform the set of class-agnostic boxes into class-specific boxes
#     by applying the predicted offsets (box_deltas)
#     """
#     if boxes.shape[0] == 0:
#         return np.zeros((0, box_deltas.shape[1]))

#     boxes = boxes.astype(np.float, copy=False)
#     widths = boxes[:, 2] - boxes[:, 0] + cfg.EPS
#     heights = boxes[:, 3] - boxes[:, 1] + cfg.EPS
#     ctr_x = boxes[:, 0] + 0.5 * widths
#     ctr_y = boxes[:, 1] + 0.5 * heights

#     dx = box_deltas[:, 0::4]
#     dy = box_deltas[:, 1::4]
#     dw = box_deltas[:, 2::4]
#     dh = box_deltas[:, 3::4]

#     pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
#     pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
#     pred_w = np.exp(dw) * widths[:, np.newaxis]
#     pred_h = np.exp(dh) * heights[:, np.newaxis]

#     pred_boxes = np.zeros(box_deltas.shape)
#     # x1
#     pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
#     # y1
#     pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
#     # x2
#     pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
#     # y2
#     pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

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

def _foward_im_roi(net, im, roi):
    """Run forward pass and return scores given boxes."""
    h_im = im.shape[0]
    w_im = im.shape[1]

    boxes = roi['boxes']
    scores = roi['scores']

    num_boxes = boxes.shape[0]
    if cfg.USE_CCL:
        im_blob_h = np.zeros((num_boxes, 3, 419, 419), dtype=np.float32)
        im_blob_o = np.zeros((num_boxes, 3, 419, 419), dtype=np.float32)
    else:
        im_blob_h = np.zeros((num_boxes, 3, 227, 227), dtype=np.float32)
        im_blob_o = np.zeros((num_boxes, 3, 227, 227), dtype=np.float32)
    if cfg.USE_SCENE:
        im_blob_s = np.zeros((num_boxes, 3, 227, 227), dtype=np.float32)
    if cfg.USE_SPATIAL > 0:
        im_blob_sr = np.zeros((num_boxes, 2, 64, 64), dtype=np.float32)
    if cfg.SHARE_O:
        score_o_blob = np.zeros((num_boxes, 1), dtype=np.float32)
    # if cfg.SHARE_V:
    #     # no additional blobs needed
    if cfg.USE_UNION:
        im_blob_ho = np.zeros((num_boxes, 3, 227, 227), dtype=np.float32)

    for i in xrange(num_boxes):
        box_h = boxes[i, 0:4]
        box_o = boxes[i, 4:8]
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
        im_blob_h[i, :, :, :] = blob_h[None, :]
        im_blob_o[i, :, :, :] = blob_o[None, :]
        if cfg.USE_SCENE:
            box_s = np.array((0, 0, w_im-1, h_im-1), dtype='uint16')
            blob_s = _get_one_blob(im, box_s, 227, 227)
            im_blob_s[i, :, :, :] = blob_s[None, :]
        if cfg.USE_SPATIAL > 0:
            if cfg.USE_SPATIAL == 1:
                # do not keep aspect ratio
                blob_sr = hdl_sr.get_map_no_pad(box_h, box_o, 64)
            if cfg.USE_SPATIAL == 2:
                # keep aspect ratio
                blob_sr = hdl_sr.get_map_pad(box_h, box_o, 64)
            im_blob_sr[i, :, :, :] = blob_sr[None, :]
        if cfg.SHARE_O:
            # use natural log of object detection scores
            score_o = np.log(scores[i, 1])
            score_o_blob[i, :] = score_o
        # if cfg.SHARE_V:
        #     # no additional blobs needed
        if cfg.USE_UNION:
            box_ho = _get_union_bbox(box_h, box_o)
            blob_ho = _get_one_blob(im, box_ho, 227, 227)
            im_blob_ho[i, :, :, :] = blob_ho[None, :]

    blobs = {'data_h': im_blob_h,
             'data_o': im_blob_o}
    if cfg.USE_SCENE:
        blobs['data_s'] = im_blob_s
    if cfg.USE_SPATIAL > 0:
        blobs['data_sr'] = im_blob_sr
    if cfg.SHARE_O:
        blobs['score_o'] = score_o_blob
    # if cfg.SHARE_V:
    #     # no additional blobs needed
    if cfg.USE_UNION:
        blobs = {'data_ho': im_blob_ho}

    # reshape network inputs
    # net.blobs['data_h'].reshape(*(blobs['data_h'].shape))
    # net.blobs['data_o'].reshape(*(blobs['data_o'].shape))
    # if cfg.USE_SCENE:
    #     net.blobs['data_s'].reshape(*(blobs['data_s'].shape))
    # if cfg.USE_SPATIAL:
    #     net.blobs['data_sr'].reshape(*(blobs['data_sr'].shape))
    # if cfg.SHARE_O:
    #     net.blobs['score_o'].reshape(*(blobs['score_o'].shape))
    # # if cfg.SHARE_V:
    # #     # no additional blobs needed
    for key in blobs:
        net.blobs[key].reshape(*(blobs[key].shape))

    blobs_out = net.forward(**(blobs))

    probs = net.blobs['cls_prob'].data

    return probs

def im_detect(net, im, roi):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        roi (dictionary):
            roi_fg ('list'): N_fg entries containing fg object boxes
            roi_bg ('list'): N_bg entries containing bg object boxes
                 boxes (ndarray): R x 8 array of humna-object proposals

    Returns:
        det_fg (list):
        det_bg (list):
            obj_id (int): object class id
            boxes (ndarray): boxes: R x 8 array of humna-object proposals
            scores (ndarray): R x K_o array of HOI class scores. K_o is the
                number of interactions classes for object o.
        obj_hoi_int (list): N entries of HOI index intervals for each object
            class. N is the number of object classes (N includes background as
            object category 0)
    """
    # roi_fg = roi['roi_fg']
    # roi_bg = roi['roi_bg']

    # det_fg = []
    # det_bg = []

    # for i in xrange(len(roi_fg)):
    #     scores = _foward_im_boxes(net, im, roi_fg[i]['boxes'])
    #     det_fg.append({'obj_id' : roi_fg[i]['obj_id'],
    #                    'boxes' : roi_fg[i]['boxes'],
    #                    'scores' : scores})

    # for i in xrange(len(roi_bg)):
    #     scores = _foward_im_boxes(net, im, roi_bg[i]['boxes'])
    #     det_bg.append({'obj_id' : roi_bg[i]['obj_id'],
    #                    'boxes' : roi_bg[i]['boxes'],
    #                    'scores' : scores})

    # return {'det_fg' : det_fg, 'det_bg' : det_bg}

    return _foward_im_roi(net, im, roi)
    

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox_h = dets[i, 0:4]
        bbox_o = dets[i, 4:8]
        ctr_x_h = (dets[i, 0] + dets[i,2]) / 2
        ctr_y_h = (dets[i, 1] + dets[i,3]) / 2
        ctr_x_o = (dets[i, 4] + dets[i,6]) / 2
        ctr_y_o = (dets[i, 5] + dets[i,7]) / 2
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox_h[0], bbox_h[1]),
                              bbox_h[2] - bbox_h[0],
                              bbox_h[3] - bbox_h[1], fill=False,
                              edgecolor='b', linewidth=3)
                )
            plt.gca().add_patch(
                plt.Rectangle((bbox_o[0], bbox_o[1]),
                              bbox_o[2] - bbox_o[0],
                              bbox_o[3] - bbox_o[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

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

def test_net_hico(net, imdb, obj_id):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    assert obj_id is not None
    hoi_ind_int = imdb.get_obj_hoi_int()[obj_id]
    num_classes = hoi_ind_int[1] - hoi_ind_int[0] + 1
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    # max_per_set = 40 * num_images
    max_per_set = 20000
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in xrange(num_classes)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = np.empty((num_classes, num_images), dtype=object)

    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    fg_obj_id = imdb.get_fg_obj_id()

    roidb = imdb.roidb
    for i in xrange(num_images):
        # read image
        im = cv2.imread(imdb.image_path_at(i))
        # get roi for the target object class
        if obj_id in fg_obj_id[i]:
            roi = [roi for roi in roidb[i]['roi_fg'] if roi['obj_id'] == obj_id]
        else:
            roi = [roi for roi in roidb[i]['roi_bg'] if roi['obj_id'] == obj_id]
        assert len(roi) == 1
        roi = roi[0]

        _t['im_detect'].tic()
        scores = im_detect(net, im, roi)
        _t['im_detect'].toc()

        _t['misc'].tic()
        # keep only the HOI scores of the target object classes
        scores = scores[:, hoi_ind_int[0]:hoi_ind_int[1]+1]
        for j in xrange(0, num_classes):
            if 'gt_classes' in roi:
                # remote GT
                inds = np.where((scores[:, j] > thresh[j]) &
                                (np.all(roi['gt_classes'] == 0, axis=1)))[0]
            else:
                inds = np.where((scores[:, j] > thresh[j]))[0]
            cls_scores = scores[inds, j]
            cls_boxes = roi['boxes'][inds, :]
            top_inds = np.argsort(-cls_scores)[:max_per_image]
            cls_scores = cls_scores[top_inds]
            cls_boxes = cls_boxes[top_inds, :]
            # push new scores onto the minheap
            for val in cls_scores:
                heapq.heappush(top_scores[j], val)
            # if we've collected more than the max number of detection,
            # then pop items off the minheap and update the class threshold
            if len(top_scores[j]) > max_per_set:
                while len(top_scores[j]) > max_per_set:
                    heapq.heappop(top_scores[j])
                thresh[j] = top_scores[j][0]

            all_boxes[j, i] = \
                    np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)

            if 0:
                class_info = imdb.classes[j+hoi_ind_int[0]]
                class_name = \
                    class_info['vname_ing'][0] + ' ' + class_info['nname'][0]
                vis_detections(im, class_name, all_boxes[j, i])
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    for j in xrange(0, num_classes):
        for i in xrange(num_images):
            inds = np.where(all_boxes[j, i][:, -1] > thresh[j])[0]
            all_boxes[j, i] = all_boxes[j, i][inds, :]

    det_file = os.path.join(output_dir, 'detections_{:02d}.mat'.format(obj_id))
    sio.savemat(det_file, {'all_boxes' : all_boxes})
