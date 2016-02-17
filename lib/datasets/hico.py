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
    def __init__(self, image_set, obj_id, obj_name, root_dir,
                 ko_train, ko_test, samp_neg, cv):
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
        self._samp_neg  = samp_neg
        self._cv        = cv
        # Set cache root
        self._cache_root = osp.abspath(osp.join(root_dir, 'data', 'cache'))
        if not os.path.exists(self._cache_root):
            os.makedirs(self._cache_root)
        # Set input paths and files        
        if cv == 0:
            self._data_path = './external/hico_20150920/images/' + image_set
            # self._anno_file = './external/hico_20150920/anno.mat'
            self._anno_file = './data/data/annotation/anno_cvpr.mat'
            self._det_path  = './caches/det_base_caffenet/' + image_set
        else:
            assert(cv == 1 or cv == 2)
            if cv == 1:
                self._anno_file = './data/data/annotation/anno_cv_01.mat'
            if cv == 2:
                self._anno_file = './data/data/annotation/anno_cv_02.mat'
            self._data_path = './external/hico_20150920/images/' + 'train2015'
            self._det_path  = './caches/det_base_caffenet/' + 'train2015'
        # Set classes
        list_action = sio.loadmat(self._anno_file)['list_action']
        if obj_id == '00':
            assert(obj_name == 'all')
            assert(not self._ko_train)
            assert(not self._samp_neg)
            class_id = range(len(list_action))
            # self._classes not set
        else:
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
            anno_vb = sio.loadmat(self._anno_file)['anno_vb']
            if self._ko_train and not self._samp_neg:
                # FLAG_SHARE_VB should be kept False
                keep_id = [np.where(anno[class_id,ind] == 1)[0].size != 0 \
                         for ind in xrange(len(lsim))]
                keep_id = np.array(keep_id)
                lsim = lsim[keep_id == True]
                anno = anno[:, keep_id == True]
                anno_vb = None
            if self._samp_neg and not self._ko_train:
                # FLAG_SHARE_VB should be kept False; using SHARE_VB does not
                # replicate what we did in ICCV (since we are not using the
                # full training set.
                assert(self._ko_train == False)
                keep_id = [   np.where(anno[class_id,ind] == 1)[0].size != 0 \
                           or np.where(anno[class_id,ind] == -2)[0].size == len(class_id) \
                           for ind in xrange(len(lsim))]
                print 'num keep: {}'.format(sum(keep_id))
                keep_id = np.array(keep_id)
                lsim = lsim[keep_id == True]
                anno = anno[:, keep_id == True]
                anno_vb = None
                # hard-coded check
                # assert(sum(keep_id) == 5000)
                # num_obj = sum([np.where(anno[class_id,ind] == 1)[0].size != 0 \
                #                for ind in xrange(len(lsim))])
                # num_bg  = sum([np.where(anno[class_id,ind] == -2)[0].size != 0 \
                #                for ind in xrange(len(lsim))])
                # print 'num obj: {}'.format(num_obj)
                # print 'num bg: {}'.format(num_bg)
            if self._ko_train and self._samp_neg:
                keep_id = [   np.where(anno[class_id,ind] == 1)[0].size != 0 \
                           or np.where(anno[class_id,ind] == -2)[0].size == len(class_id) \
                           for ind in xrange(len(lsim))]
                print 'num keep (samp_neg): {}'.format(sum(keep_id))
                keep_id = np.array(keep_id)
                lsim = lsim[keep_id == True]
                anno = anno[:, keep_id == True]
                anno_vb = anno_vb[:, keep_id == True]
                keep_id = [np.where(anno_vb[class_id,ind] == 1)[0].size != 0 \
                           for ind in xrange(len(lsim))]
                print 'num keep (ko_train): {}'.format(sum(keep_id))
                keep_id = np.array(keep_id)
                lsim = lsim[keep_id == True]
                anno = anno[:, keep_id == True]
                anno_vb = anno_vb[:, keep_id == True]
        if image_set == 'test2015':
            lsim = sio.loadmat(self._anno_file)['list_test']
            anno = sio.loadmat(self._anno_file)['anno_test']
            if self._ko_test:
                keep_id = [np.where(anno[class_id,ind] == 1)[0].size != 0 \
                         for ind in xrange(len(lsim))]
                keep_id = np.array(keep_id)
                lsim = lsim[keep_id == True]
                anno = anno[:, keep_id == True]
        self._image_index = [str(im[0][0]) for im in lsim]
        self._anno = anno[class_id,:]
        if image_set == 'train2015' and anno_vb is not None:
            self._anno_vb = anno_vb[class_id,:]
        else:
            self._anno_vb = None
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
        if self._cv == 0:
            if (    (self._ko_train and not self._samp_neg and self._image_set == 'train2015')
                 or (self._ko_test  and self._image_set == 'test2015')):
                cache_file = os.path.join(self._cache_root,
                                          self.name + '_det_caffenet_roidb_ko.pkl')
            elif self._samp_neg and not self._ko_train and self._image_set == 'train2015':
                cache_file = os.path.join(self._cache_root,
                                          self.name + '_det_caffenet_roidb_samp.pkl')
            elif self._ko_train and self._samp_neg and self._image_set == 'train2015':
                cache_file = os.path.join(self._cache_root,
                                          self.name + '_det_caffenet_roidb_ko_svb.pkl')
            elif self._obj_id == '00':
                cache_file = os.path.join(self._cache_root,
                                          self.name + '_det_caffenet_roidb_single_net.pkl')
            else:
                cache_file = os.path.join(self._cache_root,
                                          self.name + '_det_caffenet_roidb.pkl')
        else:
            assert(self._ko_train == False)
            assert(self._ko_test == False)
            assert(self._samp_neg == True)
            cache_file = os.path.join(self._cache_root,
                                      self.name + '_det_caffenet_roidb_cv{0:02d}.pkl'.format(self._cv))

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
        print 'num image: {}'.format(num_images)
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
        if (    (self._ko_train and not self._samp_neg and self._image_set == 'train2015')
             or (self._ko_test  and self._image_set == 'test2015')):
            assert(np.where(labels == 1)[0].size != 0)

        # Read VB labels
        if self._image_set == 'train2015' and self._anno_vb is not None:
            labels_vb = self._anno_vb[:, ind]
            assert(np.all(labels_vb[labels_vb != 1] == 0))
        else:
            labels_vb = None

        # Read boxes
        if obj_id == '00':
            boxes_o = None
            scores_o = None
        else:
            boxes_o, scores_o = self._get_det_one_object(res, obj_id)
            print boxes_o.shape, scores_o.shape,
        boxes_h, scores_h = self._get_det_one_object(res, hmn_id)
        # print boxes_o.shape, scores_o.shape, boxes_h.shape, scores_h.shape
        print boxes_h.shape, scores_h.shape
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

        # Set cached feature files
        reg_file_o = 'caches/cache_reg_fc6_0010/' + \
                     self._obj_id + '_' + self._obj_name + '/' + \
                     self._image_set + '/' + \
                     os.path.splitext(index)[0] + '.mat'
        ctx_file_o = 'caches/cache_ctx_pool6_0010/' + \
                     self._obj_id + '_' + self._obj_name + '/' + \
                     self._image_set + '/' + \
                     os.path.splitext(index)[0] + '.mat'
        reg_file_h = 'caches/cache_reg_fc6_0010/' + \
                     '{:02d}'.format(hmn_id) + '_person/' + \
                     self._image_set + '/' + \
                     os.path.splitext(index)[0] + '.mat'
        ctx_file_h = 'caches/cache_ctx_pool6_0010/' + \
                     '{:02d}'.format(hmn_id) + '_person/' + \
                     self._image_set + '/' + \
                     os.path.splitext(index)[0] + '.mat'

        return {'boxes_o' : boxes_o, 'scores_o' : scores_o, 
                'boxes_h' : boxes_h, 'scores_h' : scores_h,
                'label' : labels, 'label_vb' : labels_vb,
                'reg_file_o' : reg_file_o, 'ctx_file_o' : ctx_file_o,
                'reg_file_h' : reg_file_h, 'ctx_file_h' : ctx_file_h,
                'flipped' : False}

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
