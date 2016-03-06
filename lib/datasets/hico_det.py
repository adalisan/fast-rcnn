# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import os
import datasets.imdb
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
from utils.cython_bbox import bbox_overlaps

class hico_det(datasets.imdb):
    def __init__(self, image_set, obj_id, obj_name):
        if obj_id is None and obj_name is None:
            # Default setting
            name = 'hico_det_' + image_set
        # else:
        #     # KO setting
        #     name = 'hico_det_' + image_set  + '_' + obj_id + '_' + obj_name
        datasets.imdb.__init__(self, name)

        self._image_set = image_set
        self._obj_id = obj_id
        self._obj_name = obj_name
        self._data_path = './external/hico_20160224_det/images/' + image_set
        self._anno_file = './external/hico_20160224_det/anno.mat'
        self._bbox_file = './external/hico_20160224_det/anno_bbox.mat'
        self._det_path = './caches/det_base_caffenet/' + image_set
        # Load annotation
        self._load_annotation()
        # Default to roidb handler
        self._roidb_handler = self.object_detection_roidb

        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def _load_annotation(self):
        list_action = sio.loadmat(self._anno_file)['list_action']
        if self._image_set == 'train2015':
            list_im = sio.loadmat(self._anno_file)['list_train']
            gt_anno = sio.loadmat(self._anno_file)['anno_train']
        if self._image_set == 'test2015':
            list_im = sio.loadmat(self._anno_file)['list_test']
            gt_anno = sio.loadmat(self._anno_file)['anno_test']
        if self._obj_id is None and self._obj_name is None:
            # Default setting
            # Get background object id for each image
            # fg_obj_id is one-based to match the cls in detection files
            fg_obj_id = [[] for _ in xrange(len(list_im))]
            index = list_im[0, 0][0]
            det_file = os.path.join(self._det_path,
                                    os.path.splitext(index)[0] + '.mat')
            assert os.path.exists(det_file), \
                'Detection file not found at: {}'.format(det_file)
            obj_classes = sio.loadmat(det_file)['cls'][0, :]
            obj_classes = [cls[0].replace(' ', '_') for cls in obj_classes]
            obj_hoi_int = [[] for _ in xrange(len(obj_classes))]
            for oid, obj_name in enumerate(obj_classes):
                action_ind = [ind for ind, act in enumerate(list_action[:, 0])
                              if act['nname'][0] == obj_name]
                fg_ind = np.any(gt_anno[action_ind, :] == 1, axis=0)
                fg_ind = np.where(fg_ind)[0]
                for ind in fg_ind:
                    fg_obj_id[ind].append(oid)
                if action_ind:
                    obj_hoi_int[oid] = [action_ind[0], action_ind[-1]]
            # Get action to object id mapping
            # hoi_obj_id is one-based to match the cls in detection files
            obj_name_to_ind = \
                dict(zip(obj_classes, xrange(len(obj_classes))))
            hoi_obj_name = [act['nname'][0] for act in list_action[:, 0]]
            hoi_obj_id = [obj_name_to_ind[name] for name in hoi_obj_name]
        # else:
        #     # KO setting
        #     action_ind = [ind for ind, act in enumerate(list_action[:, 0])
        #                   if act['nname'][0] == self._obj_name]
        #     list_action = list_action[action_ind, :]
        #     cls_offset = action_ind[0]
        #     keep_ind = np.any(gt_anno[action_ind, :] == 1, axis=0)
        #     keep_ind = np.where(keep_ind)[0]
        #     list_im = list_im[keep_ind, :]
        #     fg_obj_id = None
        #     hoi_obj_id = None
        # TODO: use obj_hoi_int instead of cls_offset and keep_ind
        #     obj_hoi_int = None
        
        self._classes = tuple([action for action in list_action[:, 0]])
        self._image_index = [str(index[0]) for index in list_im[:, 0]]

        self._fg_obj_id = fg_obj_id
        self._hoi_obj_id = hoi_obj_id
        self._obj_hoi_int = obj_hoi_int

    def get_fg_obj_id(self):
        """
        Return foreground object indices for each image
        """
        return self._fg_obj_id

    def get_hoi_obj_id(self):
        """
        Return the object index for each HOI class
        """
        return self._hoi_obj_id

    def get_obj_hoi_int(self):
        """
        Return HOI classes index interval for each object class
        """
        return self._obj_hoi_int

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_path_from_index(self._image_index[i])

    def _image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set == 'train2015':
            gt_bbox = sio.loadmat(self._bbox_file)['bbox_train']
        if self._image_set == 'test2015':
            gt_bbox = sio.loadmat(self._bbox_file)['bbox_test']
        # if self._keep_ind is not None:
        #     # KO setting
        #     gt_bbox = gt_bbox[:, self._keep_ind]
        assert gt_bbox.shape[1] == len(self._image_index)

        gt_roidb = [self._load_bbox_annotation(bbox) for bbox in gt_bbox[0, :]]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def object_detection_roidb(self):
        """
        Return the database of detection regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_det_caffenet_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set == 'train2015':
            gt_roidb = self.gt_roidb()
            det_roidb = self._load_detection_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs_hico_det(gt_roidb, det_roidb)
        if self._image_set == 'test2015':
            roidb = self._load_detection_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_detection_roidb(self, gt_roidb):
        det_list = [self._load_detection_image(ind)
                    for ind in xrange(self.num_images)]
        return self.create_roidb_from_det_list(det_list, gt_roidb)

    def _load_detection_image(self, ind):
        # Load detection file
        index = self._image_index[ind]
        det_file = os.path.join(self._det_path,
                                os.path.splitext(index)[0] + '.mat')
        print 'Loading detection: {}'.format(index)
        assert os.path.exists(det_file), \
               'Detection file not found at: {}'.format(det_file)
        res = sio.loadmat(det_file)

        # load boxes and scores
        roi_fg = []
        roi_bg = []
        if self._obj_id is None and self._obj_name is None:
            # Default setting
            hmn_id = 1
            for obj_id in xrange(res['cls'].size):
                if obj_id == 0:
                    # skip background
                    continue
                det = self._load_detection_pair(res, hmn_id, obj_id)
                if obj_id in self._fg_obj_id[ind]:
                    roi_fg.append({'obj_id' : obj_id,
                                   'boxes' : det['boxes'],
                                   'scores' : det['scores']})
                else:
                    roi_bg.append({'obj_id' : obj_id,
                                   'boxes' : det['boxes'],
                                   'scores' : det['scores']})
        # else:
        #     # KO setting
        #     hmn_id = 1
        #     obj_id = self._obj_id
        #     det = self._load_detection_pair(res, hmn_id, obj_id)
        #     oid_fg = self._obj_id * np.ones(det['boxes'].shape[0])
        #     oid_bg = None
        #     boxes_fg = det['boxes']
        #     boxes_bg = None
        #     scores_fg = det['scores']
        #     scores_bg = None

        return {'roi_fg' : roi_fg, 'roi_bg' : roi_bg}

    def _load_detection_pair(self, res, hmn_id, obj_id):
         # Read boxes
        boxes_h, scores_h = self._get_detection_one_object(res, hmn_id)
        boxes_o, scores_o = self._get_detection_one_object(res, obj_id)

        # Keep the top 10 detection boxes; HARD-CODED thresholding here.
        top_k = 10
        boxes_h = boxes_h[0:min(top_k, boxes_h.shape[0]), :]
        boxes_o = boxes_o[0:min(top_k, boxes_o.shape[0]), :]
        scores_h = scores_h[0:min(top_k, scores_h.shape[0])]
        scores_o = scores_o[0:min(top_k, scores_o.shape[0])]

        # Generate human-object box pairs
        if hmn_id == obj_id:
            num_boxes = boxes_h.shape[0] ** 2 - boxes_h.shape[0]
        else:
            num_boxes = boxes_h.shape[0] * boxes_o.shape[0]
        count = 0
        boxes = np.zeros((num_boxes, 8), dtype=np.uint16)
        scores = np.zeros((num_boxes, 2), dtype=np.float32)
        for i in xrange(boxes_h.shape[0]):
            for j in xrange(boxes_o.shape[0]):
                if obj_id == hmn_id and i == j:
                    # Avoid the identical person in a person-person pair
                    continue
                boxes[count, 0:4] = boxes_h[i, :]
                boxes[count, 4:8] = boxes_o[j, :]
                scores[count, 0] = scores_h[i]
                scores[count, 1] = scores_o[j]
                count += 1

        return {'boxes' : boxes, 'scores' : scores}

    def _get_detection_one_object(self, res, obj_id):
        # Get detection of one object
        dets = res['dets'][0, obj_id]
        # NMS: 'keep' will also sort the dets by detection scores
        keep = np.squeeze(res['keep'][0, obj_id])
        dets = dets[keep, :]
        # Keep all the detection boxes now and filter later in data fetching
        boxes = dets[:, 0:4]
        boxes = np.around(boxes).astype('uint16')
        # Get scores
        scores = dets[:, 4].astype('float32')
        # return
        return boxes, scores

    def _load_bbox_annotation(self, gt_bbox):
        """
        Load bounding boxe pairs.
        """
        print 'Loading GT: {}'.format(gt_bbox['filename'][0])

        gt_roi = []
        # Load hoi pair bounding boxes into a data frame.
        for hoi in gt_bbox['hoi'][0, :]:
            if hoi['invis'][0, 0] == 1:
                # Ignore invisible instances
                continue
            # Get class label and convert to intra-object indices
            cls = hoi['id'][0, 0] - 1
            # if self._cls_offset is None:
            #     obj_id = self._hoi_obj_id[cls]
            # else:
            #     cls -= self._cls_offset
            #     # Skip if the class label is for a different object category
            #     if cls < 0 or cls >= self.num_classes:
            #         continue
            #     obj_id = self._obj_id
            obj_id = self._hoi_obj_id[cls]
            hoi_int = self._obj_hoi_int[obj_id]
            num_classes = hoi_int[1] - hoi_int[0] + 1
            cls -= self._obj_hoi_int[obj_id][0]
            assert cls >= 0 and cls < num_classes
            # get roi index
            rid = [ind for ind, roi in enumerate(gt_roi)
                   if obj_id == roi['obj_id']]
            assert len(rid) == 0 or len(rid) == 1
            if len(rid) == 0:
                rid = len(gt_roi)
                boxes = np.zeros((0, 8), dtype=np.uint16)
                gt_classes = np.zeros((0, num_classes), dtype=np.int32)
                overlaps = np.zeros((0, num_classes), dtype=np.float32)
            else:
                rid = rid[0]
                boxes = gt_roi[rid]['boxes']
                gt_classes = gt_roi[rid]['gt_classes']
                overlaps = gt_roi[rid]['gt_overlaps']
            # Append boxes
            for j in xrange(hoi['connection'].shape[0]):
                box_id_1 = hoi['connection'][j, 0] - 1
                box_id_2 = hoi['connection'][j, 1] - 1
                # Make pixel indexes 0-based
                x1_h = hoi['bboxhuman'][0, box_id_1]['x1'][0, 0] - 1
                y1_h = hoi['bboxhuman'][0, box_id_1]['y1'][0, 0] - 1
                x2_h = hoi['bboxhuman'][0, box_id_1]['x2'][0, 0] - 1
                y2_h = hoi['bboxhuman'][0, box_id_1]['y2'][0, 0] - 1
                x1_o = hoi['bboxobject'][0, box_id_2]['x1'][0, 0] - 1
                y1_o = hoi['bboxobject'][0, box_id_2]['y1'][0, 0] - 1
                x2_o = hoi['bboxobject'][0, box_id_2]['x2'][0, 0] - 1
                y2_o = hoi['bboxobject'][0, box_id_2]['y2'][0, 0] - 1
                box = [[x1_h, y1_h, x2_h, y2_h, x1_o, y1_o, x2_o, y2_o]]
                box = np.array(box, dtype=np.uint16)
                gt = np.zeros((1, num_classes), dtype=np.int32)
                ov = np.zeros((1, num_classes), dtype=np.float32)
                gt[0, cls] = 1.0
                ov[0, cls] = 1.0
                if True:
                    # Generate exhaustive labeling for each bounding box pair;
                    # hard-coded the ov threshold 0.5 here
                    ov_th = 0.5
                    pre_ov_h = bbox_overlaps(boxes[:, 0:4].astype(np.float),
                                             box[:, 0:4].astype(np.float))
                    pre_ov_o = bbox_overlaps(boxes[:, 4:8].astype(np.float),
                                             box[:, 4:8].astype(np.float))
                    # pre_ov = (pre_ov_h + pre_ov_o) / 2
                    # pre_ov[pre_ov_h == 0] = 0
                    # pre_ov[pre_ov_o == 0] = 0
                    pre_ov = np.minimum(pre_ov_h, pre_ov_o)
                    pre_ov = pre_ov[:, 0]
                    # previous
                    pre_ind = np.logical_and(pre_ov >= ov_th,
                                             gt_classes[:, cls] == 0)
                    pre_ind = np.where(pre_ind)[0]
                    gt_classes[pre_ind, cls] = -1.0
                    overlaps[:, cls] = np.maximum(overlaps[:, cls], pre_ov)
                    # current
                    for k in xrange(boxes.shape[0]):
                        pre_cls = np.where(gt_classes[k,:] == 1.0)[0]
                        assert pre_cls.size == 1
                        pre_cls = pre_cls[0]
                        if pre_ov[k] >= ov_th and gt[0, pre_cls] == 0:
                            gt[0, pre_cls] = -1.0
                        ov[0, pre_cls] = np.maximum(ov[0, pre_cls], pre_ov[k])
                # Concat one pair
                boxes = np.vstack((boxes, box))
                gt_classes = np.vstack((gt_classes, gt))
                overlaps = np.vstack((overlaps, ov))
                if rid >= len(gt_roi):
                    gt_roi.append({'obj_id' : obj_id,
                                   'boxes' : boxes,
                                   'gt_classes' : gt_classes,
                                   'gt_overlaps' : overlaps})
                else:
                    gt_roi[rid]['boxes'] = boxes
                    gt_roi[rid]['gt_classes'] = gt_classes
                    gt_roi[rid]['gt_overlaps'] = overlaps
        if True:
            # Change -1.0 to 1.0 for exhausive labeling
            for roi in gt_roi:
                roi['gt_classes'][roi['gt_classes'] == -1] = 1

        for roi in gt_roi:
            roi['gt_overlaps'] = scipy.sparse.csr_matrix(roi['gt_overlaps'])

        return {'roi' : gt_roi, 'flipped' : False}

if __name__ == '__main__':
    raise NotImplementedError
    # d = datasets.pascal_voc('trainval', '2007')
    # res = d.roidb
    # from IPython import embed; embed()
