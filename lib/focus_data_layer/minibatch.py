# --------------------------------------------------------
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Yu-Wei Chao
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
# import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
import os
import scipy.io as sio
# from utils.timer import Timer
import h5py

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # # Sample random scales to use for each image in this batch
    # random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
    #                                 size=num_images)
    # assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    #     'num_images ({}) must divide BATCH_SIZE ({})'. \
    #     format(num_images, cfg.TRAIN.BATCH_SIZE)
    # rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    # fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # assertions
    assert(cfg.FLAG_HICO)
    assert(cfg.FLAG_FOCUS)
    assert(cfg.FLAG_SIGMOID)
    assert(not cfg.TRAIN.BBOX_REG)
    assert(num_images > 0)
    if cfg.FLAG_HO:
        assert('boxes' not in roidb[0])
        assert(('boxes_o' in roidb[0]) and ('boxes_h' in roidb[0]))
    else:
        assert('boxes' in roidb[0])
        assert(('boxes_o' not in roidb[0]) and ('boxes_h' not in roidb[0]))

    # # Get the input image blob, formatted for caffe
    # im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    # Now, build the region of interest and label blobs
    # rois_blob = np.zeros((0, 5), dtype=np.float32)
    assert(roidb[0]['label'].shape[0] == num_classes)
    labels_blob = np.zeros((0, roidb[0]['label'].shape[0]), dtype=np.float32)
    if cfg.FLAG_SHARE_VB:
        assert(roidb[0]['label_vb'].shape[0] == num_classes)
        labels_vb_blob = np.zeros((0, roidb[0]['label_vb'].shape[0]),
                                  dtype=np.float32)
    # bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
    # bbox_loss_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
    # all_overlaps = []

    # Initalize im_blobs
    # channel_swap = (0, 3, 1, 2)
    # im_blob = np.zeros((num_images, cfg.FOCUS_H, cfg.FOCUS_W, 3), 
    #                    dtype=np.float32)
    # im_blob = im_blob.transpose(channel_swap)
    if cfg.FEAT_TYPE == 4:
       ffactor = 4
    else:
       ffactor = 1
       assert(cfg.FEAT_TYPE == 0)

    if cfg.USE_CACHE:
        assert(cfg.FLAG_HO)
        if cfg.FLAG_CTX8:
            im_blobs_o = [np.zeros((num_images, 4096, 3, 3), dtype=np.float32)
                          for _ in xrange(cfg.OBJ_K)]
            im_blobs_h = [np.zeros((num_images, 4096, 3, 3), dtype=np.float32)
                          for _ in xrange(cfg.HMN_K)]
        else:
            im_blobs_o = [np.zeros((num_images, 4096), dtype=np.float32)
                          for _ in xrange(cfg.OBJ_K)]
            im_blobs_h = [np.zeros((num_images, 4096), dtype=np.float32)
                          for _ in xrange(cfg.HMN_K)]
        if cfg.FLAG_FULLIM:
            im_blobs_s = np.zeros((num_images, 4096), dtype=np.float32)
    else:
        if cfg.FLAG_HO:
            if cfg.MODE_OBJ == -1 and cfg.MODE_HMN == -1:
                # TODO: add feat4
                # im_blobs_o = [im_blob.copy()] * cfg.OBJ_K
                # im_blobs_h = [im_blob.copy()] * cfg.HMN_K
                if cfg.FLAG_CTX8:
                    LEN_H = cfg.FOCUS_LEN_HO
                    LEN_W = cfg.FOCUS_LEN_HO
                else:
                    LEN_H = cfg.FOCUS_H
                    LEN_W = cfg.FOCUS_W
                im_blobs_o = [np.zeros((num_images, 3, LEN_H, LEN_W), dtype=np.float32)
                              for _ in xrange(cfg.OBJ_K)]
                im_blobs_h = [np.zeros((num_images, 3, LEN_H, LEN_W), dtype=np.float32)
                              for _ in xrange(cfg.HMN_K)]
            else:
                # object stream
                if cfg.MODE_OBJ == 0:
                    im_blobs_o = [np.zeros((num_images, 3, 227, 227), dtype=np.float32)
                              for _ in xrange(cfg.OBJ_K)]
                if cfg.MODE_OBJ == 1 or cfg.MODE_OBJ == 2:
                    im_blobs_o = [np.zeros((num_images, 3, 419, 419), dtype=np.float32)
                              for _ in xrange(cfg.OBJ_K)]
                # human stream
                if cfg.MODE_HMN == 0:
                    im_blobs_h = [np.zeros((num_images, 3, 227, 227), dtype=np.float32)
                              for _ in xrange(cfg.HMN_K)]
                if cfg.MODE_HMN == 1 or cfg.MODE_HMN == 2 or cfg.MODE_HMN == 6:
                    im_blobs_h = [np.zeros((num_images, 3, 419, 419), dtype=np.float32)
                              for _ in xrange(cfg.HMN_K)]
                if cfg.MODE_HMN == 3:
                    im_blobs_h = [np.zeros((num_images, 16, 64, 64), dtype=np.float32)
                              for _ in xrange(cfg.HMN_K)]
                if cfg.MODE_HMN == 4:
                    im_blobs_h = [np.zeros((num_images, 256, 16, 16), dtype=np.float32)
                              for _ in xrange(cfg.HMN_K)]
                if cfg.MODE_HMN == 5:
                    im_blobs_h = [np.zeros((num_images, 512, 8, 8), dtype=np.float32)
                              for _ in xrange(cfg.HMN_K)]
                if cfg.MODE_HMN == 6 or cfg.MODE_HMN == 7:
                    im_blobs_p = [np.zeros((num_images, 512, 8, 8), dtype=np.float32)
                              for _ in xrange(cfg.HMN_K)]
        else:
            # TODO: add ctx8
            # im_blobs = [im_blob.copy()] * cfg.TOP_K
            im_blobs = [np.zeros((num_images, 3, cfg.FOCUS_H, cfg.FOCUS_W), 
                                 dtype=np.float32) 
                        for _ in xrange(cfg.TOP_K * ffactor)]
        if cfg.FLAG_FULLIM:
            im_blobs_s = np.zeros((num_images, 3, cfg.FOCUS_H, cfg.FOCUS_W),
                                  dtype=np.float32)
    
    # timer = Timer()
    # tt = 0
    for im_i in xrange(num_images):
        if cfg.USE_CACHE:
            # TODO: add MODE_OBJ & MODE_HMN
            assert(cfg.FLAG_HO)
            if cfg.FLAG_CTX8:
                ld_o = sio.loadmat(roidb[im_i]['ctx_file_o'])
                ld_h = sio.loadmat(roidb[im_i]['ctx_file_h'])
                feat_det_o  = ld_o['feat_det_pre_ctx']
                feat_det_h  = ld_h['feat_det_pre_ctx']
                boxes_det_o = ld_o['boxes_det']
                boxes_det_h = ld_h['boxes_det']
            else:
                # timer.tic()
                ld_o = sio.loadmat(roidb[im_i]['reg_file_o'])
                ld_h = sio.loadmat(roidb[im_i]['reg_file_h'])
                # ld_o = sio.loadmat('/tmp/ywchao_job/' + roidb[im_i]['reg_file_o'][7:])
                # ld_h = sio.loadmat('/tmp/ywchao_job/' + roidb[im_i]['reg_file_h'][7:])
                # tt = tt + timer.toc()
                # timer.tic()
                feat_det_o  = ld_o['feat_det_pre_reg']
                feat_det_h  = ld_h['feat_det_pre_reg']
                boxes_det_o = ld_o['boxes_det']
                boxes_det_h = ld_h['boxes_det']
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
            for ind in xrange(cfg.OBJ_K):
                assert(np.all(roidb[im_i]['boxes_o'][ind,:] == boxes_det_o[ind,:]))
                if cfg.FLAG_CTX8:
                    im_blobs_o[ind][im_i,:,:,:] = feat_det_o[ind,:]
                else:
                    im_blobs_o[ind][im_i,:] = feat_det_o[ind,:]
            # human det feature
            for ind in xrange(cfg.HMN_K):
                assert(np.all(roidb[im_i]['boxes_h'][ind,:] == boxes_det_h[ind,:]))
                if cfg.FLAG_CTX8:
                    im_blobs_h[ind][im_i,:,:,:] = feat_det_h[ind,:]
                else:
                    im_blobs_h[ind][im_i,:] = feat_det_h[ind,:]
            # full image feature
            if cfg.FLAG_FULLIM:
                assert(np.all(feat_full_o == feat_full_h))
                im_blobs_s[im_i,:] = feat_full_o
            # tt = tt + timer.toc()
        else:
            # labels, overlaps, im_rois, bbox_targets, bbox_loss \
            #     = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
            #                    num_classes)
            # timer.tic()
            # im_file = '/tmp/ywchao_job/' + \
            #           '/'.join(roidb[im_i]['image'].split('/')[2:])
            # im = cv2.imread(im_file)
            im = cv2.imread(roidb[im_i]['image'])
            # tt = tt + timer.toc()
            h_org = im.shape[0]
            w_org = im.shape[1]
            # print roidb[im_i]['image']
            if roidb[im_i]['flipped']:
                im = im[:, ::-1, :]
            if cfg.FLAG_HO:
                if cfg.MODE_OBJ == -1 and cfg.MODE_HMN == -1:
                    # TODO: add FLAG_TOP_THRESH
                    # TODO: add feat4
                    for ind in xrange(cfg.OBJ_K):
                        if cfg.FLAG_CTX8:
                            boxes_o  = roidb[im_i]['boxes_o'][ind,:]
                            bbox_en  = _enlarge_bbox_ctx8(boxes_o, w_org, h_org)
                            bbox_en  = np.around(bbox_en[0,:]).astype(np.uint16)
                            im_focus = _get_one_blob(im, bbox_en,
                                                     cfg.FOCUS_LEN_HO, cfg.FOCUS_LEN_HO)
                        else:
                            im_focus = _get_one_blob(im, roidb[im_i]['boxes_o'][ind,:])
                            # im_focus, save_focus = _get_one_blob(im, roidb[im_i]['boxes_o'][ind,:])
                        im_blobs_o[ind][im_i, :, :, :] = im_focus
                        # savefile = 'test_i%d_o%d.jpg' % (im_i, ind)
                        # if not os.path.isfile(savefile):
                        #     cv2.imwrite(savefile,save_focus)
                    for ind in xrange(cfg.HMN_K):
                        if cfg.FLAG_CTX8:
                            boxes_h  = roidb[im_i]['boxes_h'][ind,:]
                            bbox_en  = _enlarge_bbox_ctx8(boxes_h, w_org, h_org)
                            bbox_en  = np.around(bbox_en[0,:]).astype(np.uint16)
                            im_focus = _get_one_blob(im, bbox_en,
                                                     cfg.FOCUS_LEN_HO, cfg.FOCUS_LEN_HO)
                        else:
                            im_focus = _get_one_blob(im, roidb[im_i]['boxes_h'][ind,:])
                            # im_focus, save_focus = _get_one_blob(im, roidb[im_i]['boxes_h'][ind,:])
                        im_blobs_h[ind][im_i, :, :, :] = im_focus
                        # savefile = 'test_i%d_h%d.jpg' % (im_i, ind)
                        # if not os.path.isfile(savefile):
                        #     cv2.imwrite(savefile,save_focus)
                else:
                    # assert(cfg.MODE_OBJ != -1 and cfg.MODE_HMN != -1)
                    for ind in xrange(cfg.OBJ_K):
                        if cfg.MODE_OBJ == 0:
                            im_focus = _get_one_blob(im, roidb[im_i]['boxes_o'][ind,:])
                        if cfg.MODE_OBJ == 1 or cfg.MODE_OBJ == 2:
                            boxes_o  = roidb[im_i]['boxes_o'][ind,:]
                            bbox_en  = _enlarge_bbox_ctx8(boxes_o, w_org, h_org)
                            bbox_en  = np.around(bbox_en[0,:]).astype(np.uint16)
                            im_focus = _get_one_blob(im, bbox_en, 419, 419)
                        im_blobs_o[ind][im_i, :, :, :] = im_focus
                    for ind in xrange(cfg.HMN_K):
                        if cfg.MODE_HMN == 0:
                            im_focus = _get_one_blob(im, roidb[im_i]['boxes_h'][ind,:])
                        if cfg.MODE_HMN == 1 or cfg.MODE_HMN == 2 or \
                           cfg.MODE_HMN == 6 or cfg.MODE_HMN == 7:
                            boxes_h  = roidb[im_i]['boxes_h'][ind,:]
                            bbox_en  = _enlarge_bbox_ctx8(boxes_h, w_org, h_org)
                            bbox_en  = np.around(bbox_en[0,:]).astype(np.uint16)
                            im_focus = _get_one_blob(im, bbox_en, 419, 419)
                        if cfg.MODE_HMN == 3:
                            feat_dir = 'caches/cache_pose_hmap/train2015/'
                        if cfg.MODE_HMN == 4:
                            feat_dir = 'caches/cache_pose_mid/train2015/'
                        if cfg.MODE_HMN == 5 or cfg.MODE_HMN == 6 or cfg.MODE_HMN == 7:
                            feat_dir = 'caches/cache_pose_feat_pool_1_8/train2015/'
                        if cfg.MODE_HMN == 3 or cfg.MODE_HMN == 4 or cfg.MODE_HMN == 5 or \
                           cfg.MODE_HMN == 6 or cfg.MODE_HMN == 7:
                            # load feature file
                            im_name = os.path.basename(roidb[im_i]['reg_file_h'])
                            feat_name = im_name.replace('.mat','.hdf5')
                            feat_file = feat_dir + feat_name
                            f = h5py.File(feat_file, 'r')
                            if cfg.MODE_HMN == 3 or cfg.MODE_HMN == 4 or cfg.MODE_HMN == 5:
                                im_focus = f['feat'][:][ind,:]
                            if cfg.MODE_HMN == 6 or cfg.MODE_HMN == 7:
                                im_focus_p = f['feat'][:][ind,:]
                            # assertion
                            boxes_h_1 = f['boxes'][:][ind,:]  # type float32
                            boxes_h_2 = roidb[im_i]['boxes_h'][ind,:].astype('float32')
                            diff = np.abs(boxes_h_1 - boxes_h_2)
                            assert(np.all(diff <= 1))
                            f.close()
                        im_blobs_h[ind][im_i, :, :, :] = im_focus
                        if cfg.MODE_HMN == 6 or cfg.MODE_HMN == 7:
                            im_blobs_p[ind][im_i, :, :, :] = im_focus_p
            else:
                # TODO: add ctx8
                for ind in xrange(cfg.TOP_K):
                    # Now we just take the top K detection bbox; should consider
                    # sampling K bbox from a larger pool later
                    if cfg.FLAG_TOP_THRESH:
                        # we threshold the boxes by scores
                        keep = np.where(roidb[im_i]['scores'] >= cfg.TOP_THRESH)
                        if keep[0].size == 0:
                            xid = 1
                        else:
                            xid = np.amax(keep[0]) + 1
                        pid = ind % xid
                    else:
                        pid = ind
                    # adjust boxes by feature type
                    if cfg.FEAT_TYPE == 4:
                        box_l, box_t, box_r, box_b \
                            = _get_4_side_bbox(roidb[im_i]['boxes'][pid,:], 
                                               w_org, 
                                               h_org)
                        # round box and convert type to uint16
                        box_l = np.around(box_l[0,:]).astype(np.uint16)
                        box_t = np.around(box_t[0,:]).astype(np.uint16)
                        box_r = np.around(box_r[0,:]).astype(np.uint16)
                        box_b = np.around(box_b[0,:]).astype(np.uint16)
                        im_blobs[ind*4+0][im_i, :, :, :] = _get_one_blob(im, box_l)
                        im_blobs[ind*4+1][im_i, :, :, :] = _get_one_blob(im, box_t)
                        im_blobs[ind*4+2][im_i, :, :, :] = _get_one_blob(im, box_r)
                        im_blobs[ind*4+3][im_i, :, :, :] = _get_one_blob(im, box_b)
                    else:
                        assert(cfg.FEAT_TYPE == 0)
                        im_focus = _get_one_blob(im, roidb[im_i]['boxes'][pid,:])
                        im_blobs[ind][im_i, :, :, :] = im_focus
            # full image feature
            if cfg.FLAG_FULLIM:
                box_f = np.array((0,0,w_org-1,h_org-1),dtype='uint16')
                im_blobs_s[im_i, :, :, :] = _get_one_blob(im, box_f)

        labels = roidb[im_i]['label']
        if cfg.FLAG_SHARE_VB:
            labels_vb = roidb[im_i]['label_vb']

        # # Add to RoIs blob
        # rois = _project_im_rois(im_rois, im_scales[im_i])
        # batch_ind = im_i * np.ones((rois.shape[0], 1))
        # rois_blob_this_image = np.hstack((batch_ind, rois))
        # rois_blob = np.vstack((rois_blob, rois_blob_this_image))

        # Add to labels, bbox targets, and bbox loss blobs
        labels_blob = np.vstack((labels_blob, labels.T))
        if cfg.FLAG_SHARE_VB:
            labels_vb_blob = np.vstack((labels_vb_blob, labels_vb.T))
        # bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
        # bbox_loss_blob = np.vstack((bbox_loss_blob, bbox_loss))
        # all_overlaps = np.hstack((all_overlaps, overlaps))

    # print '  minibatch: ' + str(tt)
    # For debug visualizations
    # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)

    # blobs = {'data': im_blob,
    #          'rois': rois_blob,
    #          'labels': labels_blob}
    blobs = {'labels': labels_blob}
    if cfg.FLAG_SHARE_VB:
        blobs['labels_vb'] = labels_vb_blob
    if cfg.USE_CACHE:
        assert(cfg.FLAG_HO)
        if cfg.FLAG_CTX8:
            key_base = 'pool6'
        else:
            key_base = 'fc6'
        for ind in xrange(0,cfg.OBJ_K):
            key = key_base + '_o%d' % (ind+1)
            blobs[key] = im_blobs_o[ind]
        for ind in xrange(0,cfg.HMN_K):
            key = key_base + '_h%d' % (ind+1)
            blobs[key] = im_blobs_h[ind]
        if cfg.FLAG_FULLIM:
            blobs['fc6_s'] = im_blobs_s
    else:
        if cfg.FLAG_HO:
            # TODO: add feat4
            for ind in xrange(0,cfg.OBJ_K):
                key = 'data_o%d' % (ind+1)
                blobs[key] = im_blobs_o[ind]
            for ind in xrange(0,cfg.HMN_K):
                key = 'data_h%d' % (ind+1)
                blobs[key] = im_blobs_h[ind]
                if cfg.MODE_HMN == 6:
                    key = 'data_p%d' % (ind+1)
                    blobs[key] = im_blobs_p[ind]
        else:
            for ind in xrange(0,cfg.TOP_K):
                if cfg.FEAT_TYPE == 4:
                    for i, s in enumerate(['l','t','r','b']):
                        key = 'data_%d_%s' % (ind+1,s)
                        blobs[key] = im_blobs[ind*4+i]
                else:
                    key = 'data_%d' % (ind+1)
                    blobs[key] = im_blobs[ind]
        if cfg.FLAG_FULLIM:
            blobs['data_s'] = im_blobs_s
    # if cfg.TRAIN.BBOX_REG:
    #     blobs['bbox_targets'] = bbox_targets_blob
    #     blobs['bbox_loss_weights'] = bbox_loss_blob

    return blobs

def _get_one_blob(im, bbox, len_w=None, len_h=None):
    # crop image
    # bbox indexes are zero-based
    im_focus = im[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
    # save_im  = im_focus
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
    # return im_focus, save_im

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
    # bbox_l = np.around(bbox_l).astype('uint16')
    # bbox_t = np.around(bbox_t).astype('uint16')
    # bbox_r = np.around(bbox_r).astype('uint16')
    # bbox_b = np.around(bbox_b).astype('uint16')

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

# def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
#     """Generate a random sample of RoIs comprising foreground and background
#     examples.
#     """
#     # label = class RoI has max overlap with
#     labels = roidb['max_classes']
#     overlaps = roidb['max_overlaps']
#     rois = roidb['boxes']
# 
#     # Select foreground RoIs as those with >= FG_THRESH overlap
#     fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
#     # Guard against the case when an image has fewer than fg_rois_per_image
#     # foreground RoIs
#     fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
#     # Sample foreground regions without replacement
#     if fg_inds.size > 0:
#         fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image,
#                              replace=False)
# 
#     # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
#     bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
#                        (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
#     # Compute number of background RoIs to take from this image (guarding
#     # against there being fewer than desired)
#     bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
#     bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
#                                         bg_inds.size)
#     # Sample foreground regions without replacement
#     if bg_inds.size > 0:
#         bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image,
#                              replace=False)
# 
#     # The indices that we're selecting (both fg and bg)
#     keep_inds = np.append(fg_inds, bg_inds)
#     # Select sampled values from various arrays:
#     labels = labels[keep_inds]
#     # Clamp labels for the background RoIs to 0
#     labels[fg_rois_per_this_image:] = 0
#     overlaps = overlaps[keep_inds]
#     rois = rois[keep_inds]
# 
#     bbox_targets, bbox_loss_weights = \
#             _get_bbox_regression_labels(roidb['bbox_targets'][keep_inds, :],
#                                         num_classes)
# 
#     return labels, overlaps, rois, bbox_targets, bbox_loss_weights

# def _get_image_blob(roidb, scale_inds):
#     """Builds an input blob from the images in the roidb at the specified
#     scales.
#     """
#     num_images = len(roidb)
#     processed_ims = []
#     im_scales = []
#     for i in xrange(num_images):
#         im = cv2.imread(roidb[i]['image'])
#         if roidb[i]['flipped']:
#             im = im[:, ::-1, :]
#         target_size = cfg.TRAIN.SCALES[scale_inds[i]]
#         im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
#                                         cfg.TRAIN.MAX_SIZE)
#         im_scales.append(im_scale)
#         processed_ims.append(im)
# 
#     # Create a blob to hold the input images
#     blob = im_list_to_blob(processed_ims)
# 
#     return blob, im_scales

# def _project_im_rois(im_rois, im_scale_factor):
#     """Project image RoIs into the rescaled training image."""
#     rois = im_rois * im_scale_factor
#     return rois

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

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
