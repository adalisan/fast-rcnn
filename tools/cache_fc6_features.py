#!/usr/bin/env python

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
import caffe
import argparse
import pprint
import os, sys

from utils.timer import Timer
import numpy as np
import scipy.io as sio
import cv2

# top_k = 100
# sid = 1;
# eid = 47774;

num_classes = 80
blob_name_ip = 'data'
blob_name_op = 'fc6'
w_reg = 227
h_reg = 227
w_ctx8 = 419
h_ctx8 = 419

im_base = './external/hico_20150920/images/'
anno_file = './external/hico_20150920/anno.mat'
det_base = './caches/det_base_caffenet/'


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--feat', dest='feat_root',
                        help='fc7 feature cache dir',
                        default=None, type=str)

    parser.add_argument('--top_k', type=int, required=True)
    parser.add_argument('--sid', type=int, required=True)
    parser.add_argument('--eid', type=int, required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # args = parser.parse_args('--gpu 1 \
    #                           --def ./dev_feat_extract/data/models/CaffeNet_18_horse/test.prototxt \
    #                           --net ./external/fast-rcnn/data/imagenet_models/CaffeNet.v2.caffemodel \
    #                           --top_k 100 \
    #                           --sid 1 \
    #                           --eid 47774'.split())
    args = parser.parse_args()
    return args

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
    # get enlarged bbox
    bbox_en = np.array([np.maximum(bbox[0]-0.5*r,0),
                        np.maximum(bbox[1]-0.5*r,0),
                        np.minimum(bbox[2]+0.5*r,im_width-1),
                        np.minimum(bbox[3]+0.5*r,im_height-1)])
    return bbox_en[None,:]

def _get_one_blob(im, bbox, w, h):
    # crop image
    # bbox indexes are zero-based
    im_trans = im[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
    im_trans = im_trans.astype(np.float32, copy=False)
    # subtract mean
    im_trans -= cfg.PIXEL_MEANS
    # scale
    assert(w is not None and h is not None)
    im_trans = cv2.resize(im_trans, (w, h), interpolation=cv2.INTER_LINEAR)
    # convert image to blob
    channel_swap = (2, 0, 1)
    im_trans = im_trans.transpose(channel_swap)
    return im_trans

def _get_blobs(im, rois, ctx_mode):
    num_boxes = rois.shape[0];
    if ctx_mode == 'reg':
        im_blob = np.zeros((num_boxes, 3, h_reg, w_reg), dtype=np.float32)
        for ind in xrange(num_boxes):
            im_blob[ind,:,:,:] = _get_one_blob(im, rois[ind,:], w_reg, h_reg)
    if ctx_mode == 'ctx8':
        h_ori = im.shape[0]
        w_ori = im.shape[1]
        im_blob = np.zeros((num_boxes, 3, h_ctx8, w_ctx8), dtype=np.float32)
        for ind in xrange(num_boxes):
            boxes = rois[ind,:]
            bbox_en = _enlarge_bbox_ctx8(boxes, w_ori, h_ori)
            bbox_en = np.around(bbox_en[0,:]).astype(np.uint16)
            im_blob[ind,:,:,:] = _get_one_blob(im, bbox_en, w_ctx8, h_ctx8)
    blobs = {blob_name_ip : im_blob}
    return blobs

def _get_det_one_object(res, obj_id, nms_on=True):
    # get boxes for the first object
    dets = res['dets'][0, obj_id]
    if nms_on:
        # NMS: 'keep' will also sort the dets by detection scores
        keep = np.squeeze(res['keep'][0, obj_id])
        dets = dets[keep,:]
    else:
        # sort dets by score
        order = dets[:,4].argsort()[::-1]
        dets = dets[order,:]
    # Keep all the detection boxes now and filter later in data fetching
    boxes = dets[:,0:4]
    boxes = np.around(boxes).astype('uint16')
    # get scores
    scores = dets[:,4]
    # return
    return boxes, scores


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    # get args
    top_k = args.top_k
    sid = args.sid
    eid = args.eid

    # set output path
    feat_base = './caches/cache' + \
                '_' + blob_name_op + \
                '_' + '{:04d}'.format(top_k) + \
                '/'

    # load images
    lsim_tr = sio.loadmat(anno_file)['list_train']
    lsim_ts = sio.loadmat(anno_file)['list_test']
    lsim = np.vstack((lsim_tr,lsim_ts))

    image_index = [str(im[0][0]) for im in lsim]
    num_images = len(image_index)

    # run checks
    assert(sid >= 0 and eid <= num_images)

    # timers
    _t = {'total' : Timer(), 'full' : Timer(), 'det_reg' : Timer()}

    for i in xrange(sid-1, eid):
        # set directories
        if i < lsim_tr.size:
            image_set = 'train2015'
        else:
            image_set = 'test2015'
        im_dir = im_base + image_set
        det_dir = det_base + image_set
        feat_dir = feat_base + image_set

        # make directories
        if not os.path.exists(feat_dir):
             os.makedirs(feat_dir)

        # get file names
        im_file = os.path.join(im_dir, image_index[i]);
        im_name = os.path.splitext(os.path.basename(im_file))[0]
        det_file = os.path.join(det_dir, im_name + '.mat')
        feat_file = os.path.join(feat_dir, im_name + '.mat')
        print '{:05d}/{:05d} {}'.format(i - sid + 2, eid - sid + 1, 
                                        im_name + '.jpg'),

        # skip if feature file exists
        if os.path.isfile(feat_file):
            print '\n',
            continue

        _t['total'].tic()
        # read image
        im = cv2.imread(im_file)

        # full image feature
        _t['full'].tic()
        h_ori = im.shape[0]
        w_ori = im.shape[1]
        box_f = np.array((0,0,w_ori-1,h_ori-1), dtype='uint16')
        blobs = {blob_name_ip : _get_one_blob(im, box_f, w_reg, h_reg)[None, :]}
        net.blobs[blob_name_ip].reshape(*(blobs[blob_name_ip].shape))
        blobs_out = net.forward(**(blobs))
        feat_full = np.copy(blobs_out[blob_name_op])
        _t['full'].tic()

        # detection reg feature
        _t['det_reg'].tic()
        assert os.path.exists(det_file), \
               'Detection file not found at: {}'.format(det_file)
        res = sio.loadmat(det_file)
        assert res['dets'].shape[1] == num_classes+1, \
               'Incorrect number of classes in {}'.format(det_file)
        feat_det_reg = np.empty(num_classes, dtype=object)
        # exclude background class: start index from 1
        for j in xrange(1, num_classes+1):
            # get detection boxes
            nms_on = True
            boxes, _ = _get_det_one_object(res, j, nms_on)
            # limit the number of boxes
            if boxes.shape[0] > top_k:
                boxes = boxes[0:top_k,:]
            # get input blob
            blobs = _get_blobs(im, boxes, 'reg')
            # reshape network inputs
            net.blobs[blob_name_ip].reshape(*(blobs[blob_name_ip].shape))
            # forward pass
            blobs_out = net.forward(**(blobs))
            # copy the output
            feat_det_reg[j-1] = np.copy(blobs_out[blob_name_op])
        _t['det_reg'].toc()

        # save output
        sio.savemat(feat_file, {'feat_full' : feat_full, 
                                'feat_det_reg' : feat_det_reg})
        
        _t['total'].toc()
        print 'full: {:.3f}s det_reg: {:.3f}s total: {:.3f}s' \
              .format(_t['full'].average_time, _t['det_reg'].average_time, 
                      _t['total'].average_time)

    print 'Done.'
