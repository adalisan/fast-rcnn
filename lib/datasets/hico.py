# --------------------------------------------------------
# Written by Yu-Wei Chao
# --------------------------------------------------------

import datasets
# import datasets.pascal_voc
import datasets.im_horse
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

import os.path as osp
# import cv2

class hico(datasets.imdb):
    def __init__(self, image_set, obj_id, obj_name, root_dir, ko_train, ko_test):
        # image_set: 'train2015' or 'test2015'
        #            equavilent to 'train2015_ho' and 'test2015_ho' in im_horse
        # obj_id:    '18'
        # obj_name:  'horse'
        name = 'hico_' + image_set + '_' + obj_id + '_' + obj_name
        datasets.imdb.__init__(self, name, True)
        self._image_set = image_set
        self._obj_id    = obj_id
        self._obj_name  = obj_name
        self._ko_train  = ko_train
        self._ko_test   = ko_test
        # Set cache root
        self._cache_root = osp.abspath(osp.join(root_dir, 'data', 'cache'))
        if not os.path.exists(self._cache_root):
            os.makedirs(self._cache_root)
        # Set input paths and files
        self._data_path = './external/hico_20150920/images/' + image_set
        self._anno_file = './external/hico_20150920/anno.mat'
        self._det_path  = './caches/det_base_caffenet/' + image_set
        # Set classes
        list_action = sio.loadmat(self._anno_file)['list_action']
        class_id = [idx for idx, action in enumerate(list_action) 
                    if action[0][0][0] == obj_name]
        assert(class_id)
        self._classes = [str(list_action[id][0][1][0]) for id in class_id]
        self._classes = tuple(self.classes)
        # Load image list and annotation
        assert(image_set == 'train2015' or image_set == 'test2015')
        if image_set == 'train2015':
            lsim = sio.loadmat(self._anno_file)['list_train']
            anno = sio.loadmat(self._anno_file)['anno_train']
            if self._ko_train:
                keep_id = [np.where(anno[class_id,ind] == 1)[0].size != 0 \
                         for ind in xrange(len(lsim))]
                keep_id = np.array(keep_id)
                lsim = lsim[keep_id == True]
                anno = anno[:,keep_id == True]
        if image_set == 'test2015':
            lsim = sio.loadmat(self._anno_file)['list_test']
            anno = sio.loadmat(self._anno_file)['anno_test']
            if self._ko_test:
                keep_id = [np.where(anno[class_id,ind] == 1)[0].size != 0 \
                         for ind in xrange(len(lsim))]
                keep_id = np.array(keep_id)
                lsim = lsim[keep_id == True]
                anno = anno[:,keep_id == True]
        self._image_index = [str(im[0][0]) for im in lsim]
        self._anno = anno[class_id,:]
        # Default to roidb handler
        self._roidb_handler = self.object_detection_roidb
        # Check path exists
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        # image_index contains the file names
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # index contains the file names
        image_path = os.path.join(self._data_path, index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def object_detection_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        if (    (self._ko_train and self._image_set == 'train2015')
             or (self._ko_test  and self._image_set == 'test2015')):
            cache_file = os.path.join(self._cache_root,
                                      self.name + '_det_caffenet_roidb_ko.pkl')
        else:
            cache_file = os.path.join(self._cache_root,
                                      self.name + '_det_caffenet_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        # Load detection bbox and scores
        roidb = self._load_detection_roidb()

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_detection_roidb(self):        
        # Load detection results
        num_images = len(self._image_index)
        print 'num_image: {}'.format(num_images)
        roidb = [self._load_detection(ind) for ind in xrange(num_images)]
        return roidb

    def _load_detection(self, ind):
        # TODO: handle hmn + hmn
        obj_id = self._obj_id
        hmn_id = 1

        # Load detection file
        index = self._image_index[ind]
        filename = os.path.join(self._det_path, os.path.splitext(index)[0] + '.mat')
        print 'Loading: {}'.format(index),
        assert os.path.exists(filename), \
               'Detection file not found at: {}'.format(filename)
        res = sio.loadmat(filename)
        
        # Read labels
        labels = self._anno[:, ind]
        labels[labels != 1] = 0
        if (    (self._ko_train and self._image_set == 'train2015')
             or (self._ko_test  and self._image_set == 'test2015')):
            assert(np.where(labels == 1)[0].size != 0)

        # Read boxes
        boxes_o, scores_o = self._get_det_one_object(res, obj_id)
        boxes_h, scores_h = self._get_det_one_object(res, hmn_id)
        print boxes_o.shape, scores_o.shape, boxes_h.shape, scores_h.shape
        # im = cv2.imread(self.image_path_from_index(index))
        # for ind in xrange(10):
        #     savefile = 'test_o%d.jpg' % ind
        #     if not os.path.isfile(savefile):
        #         bbox = boxes_o[ind,:]
        #         im_focus = im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        #         cv2.imwrite(savefile,im_focus)
        #     savefile = 'test_h%d.jpg' % ind
        #     if not os.path.isfile(savefile):
        #         bbox = boxes_h[ind,:]
        #         im_focus = im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        #         cv2.imwrite(savefile,im_focus)
        return {'boxes_o' : boxes_o, 'scores_o' : scores_o, 
                'boxes_h' : boxes_h, 'scores_h' : scores_h,
                'label' : labels, 'flipped' : False}

    # get boxes for object object with nms
    def _get_det_one_object(self, res, obj_id):
        # get boxes for the first object
        dets = res['dets'][0, obj_id]
        # NMS: 'keep' will also sort the dets by detection scores
        keep = np.squeeze(res['keep'][0, obj_id])
        dets = dets[keep,:]
        # Keep all the detection boxes now and filter later in data fetching
        boxes = dets[:,0:4]
        boxes = np.around(boxes).astype('uint16')
        # get scores
        scores = dets[:,4]
        # return
        return boxes, scores

    # # TODO: edit this function if neccessary
    # def _write_voc_results_file(self, all_boxes):
    #     use_salt = self.config['use_salt']
    #     comp_id = 'comp4'
    #     if use_salt:
    #         comp_id += '-{}'.format(os.getpid())

    #     # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
    #     path = os.path.join(self._devkit_path, 'results', 'VOC' + self._year,
    #                         'Main', comp_id + '_')
    #     for cls_ind, cls in enumerate(self.classes):
    #         if cls == '__background__':
    #             continue
    #         print 'Writing {} VOC results file'.format(cls)
    #         filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
    #         with open(filename, 'wt') as f:
    #             for im_ind, index in enumerate(self.image_index):
    #                 dets = all_boxes[cls_ind][im_ind]
    #                 if dets == []:
    #                     continue
    #                 # the VOCdevkit expects 1-based indices
    #                 for k in xrange(dets.shape[0]):
    #                     f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
    #                             format(index, dets[k, -1],
    #                                    dets[k, 0] + 1, dets[k, 1] + 1,
    #                                    dets[k, 2] + 1, dets[k, 3] + 1))
    #     return comp_id

    # # TODO: edit this function if neccessary
    # def _do_matlab_eval(self, comp_id, output_dir='output'):
    #     rm_results = self.config['cleanup']

    #     path = os.path.join(os.path.dirname(__file__),
    #                         'VOCdevkit-matlab-wrapper')
    #     cmd = 'cd {} && '.format(path)
    #     cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
    #     cmd += '-r "dbstop if error; '
    #     cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
    #            .format(self._devkit_path, comp_id,
    #                    self._image_set, output_dir, int(rm_results))
    #     print('Running:\n{}'.format(cmd))
    #     status = subprocess.call(cmd, shell=True)

    # # TODO: edit this function if neccessary
    # def evaluate_detections(self, all_boxes, output_dir):
    #     comp_id = self._write_voc_results_file(all_boxes)
    #     self._do_matlab_eval(comp_id, output_dir)

    # # TODO: edit this function if neccessary
    # def competition_mode(self, on):
    #     if on:
    #         self.config['use_salt'] = False
    #         self.config['cleanup'] = False
    #     else:
    #         self.config['use_salt'] = True
    #         self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.im_horse('train2015')
    res = d.roidb
    from IPython import embed; embed()
