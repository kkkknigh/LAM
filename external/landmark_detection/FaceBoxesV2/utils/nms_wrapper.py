# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

try:
    from .nms.cpu_nms import cpu_nms
except ImportError:
    from .nms.py_cpu_nms import py_cpu_nms as cpu_nms

import numpy as np 

def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    return cpu_nms(dets.astype(np.float32), float(thresh))
