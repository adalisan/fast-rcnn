# --------------------------------------------------------
# Written by Yu-Wei Chao
# --------------------------------------------------------

"""Build spatial relation maps for training HO R-CNN."""

import numpy as np

def get_map_no_pad(box1, box2, length):
    """
    Input:
        box1: 4 np array with coordinates representing the first box 
            {x11, y11, x12, y12}
        box2: 4 np array with coordinates representing the second box 
            {x21, y21, x22, y22}
        length: scaling factor
        
    Returns:
        3D np array (2 x length x length), with the first dimension representing
        the filter on human, and the second dimension representing the filter on
        object 
    """
    # get minimum x, y
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    # translate
    bx1 = np.array([box1[0] - x_min, box1[1] - y_min,
    	            box1[2] - x_min, box1[3] - y_min], dtype=np.float32)
    bx2 = np.array([box2[0] - x_min, box2[1] - y_min,
    	            box2[2] - x_min, box2[3] - y_min], dtype=np.float32)
    # get new width and height
    w = max(bx1[2], bx2[2]) - min(bx1[0], bx2[0])
    h = max(bx1[3], bx2[3]) - min(bx1[1], bx2[1])
    # scale
    factor_w = np.float(length) / np.float(w)
    factor_h = np.float(length) / np.float(h)
    bx1_rs = np.array([bx1[0] * factor_w, bx1[1] * factor_h,
                       bx1[2] * factor_w, bx1[3] * factor_h])
    bx2_rs = np.array([bx2[0] * factor_w, bx2[1] * factor_h,
                       bx2[2] * factor_w, bx2[3] * factor_h])

    # generate map
    map_1 = np.zeros([length, length], dtype=np.uint8)
    map_2 = np.zeros([length, length], dtype=np.uint8)
    for i in range(int(round(bx1_rs[1])), int(round(bx1_rs[3]))):
        for j in range(int(round(bx1_rs[0])), int(round(bx1_rs[2]))):
            map_1[i][j] = 1
    for i in range(int(round(bx2_rs[1])), int(round(bx2_rs[3]))):
        for j in range(int(round(bx2_rs[0])), int(round(bx2_rs[2]))):
            map_2[i][j] = 1
    return np.array([map_1, map_2]), bx1_rs, bx2_rs

def get_map_pad(box1, box2, length):
    """
    Input:
        box1: 4 np array with coordinates representing the first box
            {x11, y11, x12, y12}
        box2: 4 np array with coordinates representing the second box
            {x21, y21, x22, y22}
        length: size of the returning image
        
    Returns:
        3D np array (2 x length x length), with the first dimension representing
        the filter on human, and the second dimension representing the filter on
        object 
    """
    # get minimum x, y
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    # translate
    bx1 = np.array([box1[0] - x_min, box1[1] - y_min,
    	            box1[2] - x_min, box1[3] - y_min], dtype=np.float32)
    bx2 = np.array([box2[0] - x_min, box2[1] - y_min,
    	            box2[2] - x_min, box2[3] - y_min], dtype=np.float32)
    # get new width and height
    w = max(bx1[2], bx2[2]) - min(bx1[0], bx2[0])
    h = max(bx1[3], bx2[3]) - min(bx1[1], bx2[1])
    # scale
    if h > w:
        factor = np.float(length) / np.float(h)
        num_pad = int(round((length - factor * w) / 2.0))
        bx1_rs = np.array([bx1[0] * factor + num_pad, bx1[1] * factor,
                           bx1[2] * factor + num_pad, bx1[3] * factor])
        bx2_rs = np.array([bx2[0] * factor + num_pad, bx2[1] * factor,
                           bx2[2] * factor + num_pad, bx2[3] * factor])
    else:
        factor = np.float(length) / np.float(w)
        num_pad = int(round((length - factor * h) / 2.0))
        bx1_rs = np.array([bx1[0] * factor, bx1[1] * factor + num_pad,
                           bx1[2] * factor, bx1[3] * factor + num_pad])
        bx2_rs = np.array([bx2[0] * factor, bx2[1] * factor + num_pad,
                           bx2[2] * factor, bx2[3] * factor + num_pad])

    # generate map
    map_1 = np.zeros([length, length], dtype=np.uint8)
    map_2 = np.zeros([length, length], dtype=np.uint8)
    for i in range(int(round(bx1_rs[1])), int(round(bx1_rs[3]))):
        for j in range(int(round(bx1_rs[0])), int(round(bx1_rs[2]))):
            map_1[i][j] = 1
    for i in range(int(round(bx2_rs[1])), int(round(bx2_rs[3]))):
        for j in range(int(round(bx2_rs[0])), int(round(bx2_rs[2]))):
            map_2[i][j] = 1
    return np.array([map_1, map_2]), bx1_rs, bx2_rs
