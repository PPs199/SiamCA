from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from siamca.core.config import cfg
from siamca.utils.bbox import corner2center
from siamca.utils.point import Point


class PointTarget:
    def __init__(self,):
        self.points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE//2)

    def __call__(self, target, size, neg=False):

        # -1 ignore 0 negative 1 positive
        cls = -1 * np.ones((size, size), dtype=np.int64)
        delta = np.zeros((2, size, size), dtype=np.float32)
        shape = np.zeros((2, size, size), dtype=np.float32)
        # conner= np.zeros((4, size, size), dtype=np.float32)
        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        tcx, tcy, tw, th = corner2center(target)
        points = self.points.points

        if neg:
            neg = np.where(np.square(tcx - points[0]) / np.square(tw / 4) +
                           np.square(tcy - points[1]) / np.square(th / 4) < 1)
            neg, neg_num = select(neg, cfg.TRAIN.NEG_NUM)
            cls[neg] = 0
            shape[0][:][:] = 1e-7  
            shape[1][:][:] = 1e-7 
            return cls, delta,shape

        delta[0] = (tcx - points[0])
        delta[1] = (tcy - points[1])

        # ellipse label
        pos = np.where(np.square(tcx - points[0]) / np.square(tw / 4) +
                       np.square(tcy - points[1]) / np.square(th / 4) < 1)
        neg = np.where(np.square(tcx - points[0]) / np.square(tw / 2) +
                       np.square(tcy - points[1]) / np.square(th / 2) > 1)
        
        # sampling
        pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)
        neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)
        cls[pos] = 1
        cls[neg] = 0
        shape[0][:][:] = tw
        shape[1][:][:] = th
        return cls, delta, shape/64 
