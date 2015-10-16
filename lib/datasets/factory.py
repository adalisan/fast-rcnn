# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.pascal_voc
import datasets.im_horse
import numpy as np

from fast_rcnn.config import cfg
import os

def _selective_search_IJCV_top_k(split, year, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = datasets.pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year))

# Set up voc_<year>_<split>_top_<k> using selective search "quality" mode
# but only returning the first k boxes
for top_k in np.arange(1000, 11000, 1000):
    for year in ['2007', '2012']:
        for split in ['train', 'val', 'trainval', 'test']:
            name = 'voc_{}_{}_top_{:d}'.format(year, split, top_k)
            __sets[name] = (lambda split=split, year=year, top_k=top_k:
                    _selective_search_IJCV_top_k(split, year, top_k))

# Set up im_horse using CaffeNet object detector
# 1st row: horse
# 2nd row: person; both options will use sigmoid by default
# 3rd row: horse + person; both options will use sigmoid by default
hico_set = ['train2015_single', 'train2015', 'test2015', 'train2015_sigmoid',
            'train2015_person', 'test2015_person',
            'train2015_ho', 'test2015_ho']
for image_set in hico_set:
    name = 'im_horse_{}'.format(image_set)
    __sets[name] = (lambda image_set=image_set:
                    datasets.im_horse(image_set, cfg.ROOT_DIR))

# Set up hico using CaffeNet object detector
file_obj = os.path.join(cfg.ROOT_DIR, 'data/hico/list_coco_obj')
list_obj = [line.strip() for line in open(file_obj)];
hico_set = ['train2015', 'test2015']

for idx, obj_name in enumerate(list_obj):
    obj_id = '{:02d}'.format(idx+1)
    for image_set in hico_set:
        name = 'hico_' + image_set + '_' + obj_id + '_' + obj_name
        __sets[name] = (lambda image_set=image_set, obj_id=obj_id, obj_name=obj_name:
                        datasets.hico(image_set, obj_id, obj_name, cfg.ROOT_DIR))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
